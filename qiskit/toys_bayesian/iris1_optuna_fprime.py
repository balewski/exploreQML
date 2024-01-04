#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Jan:
works with EV not probs

added eps-precision
'''

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
    
import optuna
import logging
from functools import partial

from scipy.optimize import minimize, approx_fprime

# Set Optuna's log level to warning to reduce output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define a callback function to print the result every 10 trials
#...!...!....................
def print_callback(study, trial):
    if trial.number % 50 == 0:
        print('Optuna Trial=%d  loss=%.3f  (best=%.3f)'%(trial.number,trial.value, study.best_value))
        #1print(f"    Parameters: {trial.params}")
        
#...!...!....................
def optuna_objective(trial, X, Y,nW,wMax):
    # Define the range of parameters for Bayesian Optimization
    params = [trial.suggest_float(f'param_{i}', -wMax, wMax) for i in range(nW)]
    loss=loss_function(params, X, Y)
    return loss

# Optuna's stopping criteria
def early_stopping_callback(study, trial, epsilon=1e-3, n_worse_trials=5, n_burnin_trials=10):
    # Initialize or increment the worse_trials_counter
    if 'worse_trials_counter' not in study.user_attrs:
        study.set_user_attr('worse_trials_counter', 0)
    worse_trials_counter = study.user_attrs['worse_trials_counter']

    # Check if this is the best trial so far
    if study.best_value is None or trial.number <= n_burnin_trials:
        return

    if abs(study.best_value - study.best_trial.value) < epsilon:
        # The trial has not improved enough, increment the counter
        worse_trials_counter += 1
        study.set_user_attr('worse_trials_counter', worse_trials_counter)
    else:
        # The trial has improved, reset the counter
        study.set_user_attr('worse_trials_counter', 0)

    # Check if the number of worse trials has exceeded the threshold
    if worse_trials_counter >= n_worse_trials:
        study.set_user_attr('early_stopped', True)
        study.set_user_attr('worse_trials_counter', 0) # reset counter
        print('\n*** Early stop on nTrial=%d, nWorse=%d, eps=%.3f\n'%(trial.number,worse_trials_counter,epsilon))
       
        study.stop()

        
# Binary cross-entropy loss function
#...!...!....................
def binary_cross_entropy_loss(Y_pred, Y_true):
    #print('BCEL:',Y_pred.shape, Y_true.shape)
    #print('few vals Y_pred:',Y_pred[:5],'\nYtrue:',Y_true[:5]); aaa
    return -np.mean(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred + 1e-9))


# Loss function for optimization including biases
#...!...!....................
def loss_function(params, X, Y):    
    Y_pred = forward_pass(X, params)
    loss =  binary_cross_entropy_loss(Y_pred, Y)
    loss_history.append(loss)
    return loss

# Activation functions
#...!...!....................
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

#...!...!....................
def russianRulette(X,W,B,isLin=True):
    # input is in EV, not in prob
    
    if 0:
        print('\nRR shapes X,W,B:', X.shape,W.shape,B.shape)
        print('X:',X); print('W:',W); print('B:',B)
        print_range(X,'X');  print_range(W,'W');  print_range(B,'B')
        
    #.... input is in range [-1,1], as required for expectation values
    m1=X.shape[1]+1  # all inputs+bias
    nImg=X.shape[0]
    nCell=B.shape[0]
    Y=[]
    for im in range(nImg):
        xV=X[im]
        yV=[]
        for ic in range(nCell):
            wV=W[:,ic]
            b=B[ic]
            # convert all EV to probs
            pX=(1-xV)/2
            pW=(1-wV)/2
            pB=(1-b)/2
            XpW=pX*pW
            #print(im,ic,XpW.shape)
            if isLin:
                xwSum=np.sum(XpW)
                y=(xwSum+pB)/m1
            else:
                xw1Prod=np.prod(1-XpW)
                y=1 - (1-pB)*xw1Prod  # multi-Bernoullie

                if 0:  # combined OR+AND
                    xw1ProdB=np.prod(XpW)
                    y2=xw1ProdB  # AND 
                    y=(y+y2)/2.
                    #y=y2
            yV.append(y)
        Y.append(yV)
    Y=np.array(Y)
    return Y  # returns probability

# Forward pass including biases
#...!...!....................
def forward_pass(X, paramsL):
    #print('FoP:params',params)
    params=np.array(paramsL)
    params=np.tanh(params)  # rescale params to range [-1,1]
    
    if  not doStandard: # CHECK sanity of params
        pMin,pMax=np.min(params),np.max(params)
        #print('FPS p',pMin,pMax)
        assert pMin>=-1
        assert pMax<=1
        xMin,xMax=np.min(X),np.max(X)
        #print('FPS x',xMin,xMax)
        assert xMin>=-1
        assert xMax<=1

        
    # Unpack parameters
    W1 = params[:end_W1].reshape(input_size, hidden_size)
    b1 = params[end_W1:end_b1]
    W2 = params[end_b1:end_W2].reshape(hidden_size, output_size)
    b2 = params[end_W2:]

    if doStandard:
        Z1 = np.dot(X, W1) + b1
        Z2 = np.dot(Z1, W2) + b2
        A2 = softmax(Z2)
    else:
        A1 = russianRulette(X,W1,b1,isLin= not True);  # returns probability
        #A1=1-2*A1  # converts prob back to EV - degrades convergence
        A2 = russianRulette(A1,W2,b2,isLin= not True)
    return A2

#...!...!....................
def normalize_input_data(X):
    # The standard score of a sample x is calculated as:  z = (x - u) / s
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    print('NID end: Z shape:',Z.shape)
    
    if 1:
        blowFact=1.
        print('NID add tanh transform, blowFact=',blowFact)
        Z=np.tanh(Z*blowFact)
 
    print_range(Z,'inp norm')
      
    return Z

#...!...!....................
def print_range(Z,text):
    print('\ndata range (%s)  shape:'%text,Z.shape)
    print('mean :',np.mean(Z,axis=0))
    print('std  :',np.std(Z,axis=0))
    print('min  :',np.min(Z,axis=0))
    print('max  :',np.max(Z,axis=0))

#...!...!....................
def gradient(params, X, Y, epsilon=1e-4):
    # Approximate the gradient of the objective function
    #epsilon = np.sqrt(np.finfo(float).eps)  # =1.5e-8
    return approx_fprime(params, lambda p: loss_function(p, X, Y), epsilon)

#...!...!....................
def init_rnd_weights(): # not used when 'Optuna' runs 1st
    # Initialize weights    
    W1 = np.random.rand(input_size, hidden_size)
    b1 = np.random.rand(hidden_size)
    W2 = np.random.rand(hidden_size, output_size)
    b2 = np.random.rand(output_size)


    # Flatten weights and biases for optimization
    weightsIni = np.hstack((W1.ravel(), b1, W2.ravel(), b2))
    assert numWeight==weightsIni.shape[0] 
    
#...!...!....................
def predict_iris(X,Y,weights,text):
    Y_pred = forward_pass(X, weightsOpt)
    Y_pred_class = np.argmax(Y_pred, axis=1)
    Y_true_class = np.argmax(Y, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_class == Y_true_class)
    print(f"Accuracy {text}: {accuracy * 100:.2f}%")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix( Y_true_class,Y_pred_class )
    print('\nconfusion matrix, test samples:%d'%(X.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco:%s'%(i,rec))
        
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=3)
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y=np.roll(y,-1)  # re-label classes
    #np.random.shuffle(y)

    # Preprocess the data
    X_scaled =normalize_input_data(X)
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    
     
    # Neural network parameters
    input_size = X.shape[1] # input features (4)
    hidden_size = 5  # 5 is good
    output_size = y_onehot.shape[1]  # number of output classes (3)
    mxIterA=400  # for Optuna  x2 
    mxIterB=500  # for L-BFGS-B
    rnd_seed=42 # for reproducibility of data split
    min_ftol=1e-2
    
    maxWeights=3.0  #  both optimizers use wider range , then apply sigmoid
    doStandard=False # use stochastic perceptron
    #doStandard= True  # traditional ML w/ softmax activation

    assert hidden_size>=1
    np.random.seed(rnd_seed)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=rnd_seed)

    if 0:
        nTrainSamp=5
        X_train=X_train[:nTrainSamp]; y_train=y_train[:nTrainSamp]
        print('M: reduced train samples to %d'%nTrainSamp)

    print('M: train size:%s   hidden_size=%d  Optuma nIter=%d , min_ftol=%.1e'%(X_train.shape,hidden_size,mxIterA,min_ftol))

    
    # Define the end indices for each set of parameters
    end_W1 = input_size * hidden_size
    end_b1 = end_W1 + hidden_size
    end_W2 = end_b1 + hidden_size * output_size
    numWeight=end_W2+output_size
    loss_history = []
    
    if 0:  # forward pass one time
        outV=forward_pass(X_train, weightsIni)
        print('\noutV:',outV.shape, outV[:5],y_train[:5])
        exit(0)

    # Create a study object and specify the optimization direction
    study = optuna.create_study(direction='minimize')

    for m in range(2):
        study.optimize(partial(optuna_objective, X=X_train, Y=y_train,  nW=numWeight,wMax=maxWeights), n_trials=mxIterA, callbacks=[print_callback, lambda study, trial: early_stopping_callback(study, trial,  epsilon=min_ftol, n_worse_trials=50, n_burnin_trials=200)])
        print("Optimal Optuna-%d loss:%.3f"%(m, study.best_value))
   
        # Get the best parameters found by Optuna
        optParamsD = study.best_params
        weightsOpt= np.array([optParamsD[f'param_{i}'] for i in range(numWeight)])       
        print_range(weightsOpt,'weightsOpt Optuna-%d'%m)
        if m>0: print('Optima best params:',weightsOpt)
        # Test the model
        predict_iris(X_test,y_test,weightsOpt,text='phase1-%d-test'%m)
        if study.user_attrs.get('early_stopped', False):
            print("M:The study stopped, no continuation of Optuna training\n")
            break
    if 1:
        print('M: Phase 2: Gradient-Based Refinement nIter=%d'%mxIterB)
        #1 weightsOpt=weightsIni  #skip Bayesian optimizer, degrades convergence
        param_bounds = [(-maxWeights, maxWeights) for _ in range(numWeight)]  # Set bounds for each parameter
       
       

        partial_gradient = partial(gradient, X=X_train, Y=y_train, epsilon=min_ftol)
        result = minimize(fun=lambda p: loss_function(p, X_train, y_train),
                      x0=list(weightsOpt),
                      method='L-BFGS-B', bounds=param_bounds, # L-BFGS-B supports bounds
                      jac=partial_gradient,
                      options={'maxiter':mxIterB, 'ftol': min_ftol})

        # Output results
        print("Refined optimal parameters found by BFGS:", result.x)
        print("Refined best loss found by L-BFGS-B:", result.fun)

        # Extract the optimized weights
        weightsOpt = result.x
        print_range(weightsOpt,'weightsOpt L-BFGS-B+grad')
        # Test the model
        predict_iris(X_test,y_test,weightsOpt,text='phase2-test')

    print('M:done')
