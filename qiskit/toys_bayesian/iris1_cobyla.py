#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Jan:


'''

import numpy as np
from scipy.optimize import minimize
#from scipy.optimize import basinhopping
#from scipy.optimize import differential_evolution
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
                y=1 - (1-pB)*xw1Prod            
                #print(1-XpW,1-pB,xw1Prod,y)
            yV.append(y)
        Y.append(yV)
    Y=np.array(Y)
    #print('Y:',Y)
    #print_range(Y,'Y')
    return Y

# Forward pass including biases
#...!...!....................
def forward_pass(X, params):
    doStandard=  True
    params=np.tanh(params)
    if 0 and not doStandard: # CHECK sanity of params
        pMin,pMax=np.min(params),np.max(params)
        #print('FPS p',pMin,pMax)
        assert pMin>=-1
        assert pMax<=1
        #xMin,xMax=np.min(X),np.max(X)
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
        A1 = russianRulette(X,W1,b1,isLin=True);
        A2 = russianRulette(A1,W2,b2,isLin=not True)
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
    #y=np.roll(y,-1)
    #np.random.shuffle(y)

    # Preprocess the data
    X_scaled =normalize_input_data(X)
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

     
    # Neural network parameters
    input_size = X.shape[1] # input features (4)
    hidden_size = 2
    output_size = y_onehot.shape[1]  # number of categories (3)
    mxIter=800  # for COBYLA
    rnd_seed=42 # for reproducibility of data split

    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=rnd_seed)

    if 0:
        nTrainSamp=5
        X_train=X_train[:nTrainSamp]; y_train=y_train[:nTrainSamp]
        print('M: reduced train samples to %d'%nTrainSamp)


    # Initialize weights
    np.random.seed(rnd_seed)
    W1 = np.random.rand(input_size, hidden_size)
    b1 = np.random.rand(hidden_size)
    W2 = np.random.rand(hidden_size, output_size)
    b2 = np.random.rand(output_size)


    # Flatten weights and biases for optimization
    weightsIni = np.hstack((W1.ravel(), b1, W2.ravel(), b2))

    # Define the end indices for each set of parameters
    end_W1 = input_size * hidden_size
    end_b1 = end_W1 + hidden_size
    end_W2 = end_b1 + hidden_size * output_size

    # Loss history
    loss_history = []
    
    if 0:  # forward pass one time
        outV=forward_pass(X_train, weightsIni)
        print('\noutV:',outV.shape, outV[:5],y_train[:5])
        exit(0)

    # Use COBYLA optimizer with set maximum of iterations
    result = minimize(fun=loss_function, 
                      x0=weightsIni,
                      args=(X_train, y_train), 
                      method='COBYLA', 
                      options={'maxiter': mxIter,'rhobeg': 0.3})
        # 'rhobeg' controls the initial step size of the parameters when computing the next value of the function, default is 1.0. Smaller step increases number of iterations

    # Print loss history
    print("\nLoss History (Iteration vs. Loss):")
    for j, loss in enumerate(loss_history):
        if j%50==0 or j==mxIter-1:
            print(f"Iteration {j + 1}: Loss = {loss:.3f}")

    # Extract the optimized weights
    weightsOpt = result.x
    wMin=min(weightsOpt)
    wMax=max(weightsOpt)
    wAvr=np.mean(weightsOpt)

    print('num par=',weightsOpt.shape,'W in [%.2f,%.2f] avr=%.2f'%(wMin,wMax,wAvr))

    # Test the model
    Y_pred_test = forward_pass(X_test, weightsOpt)
    Y_pred_test_class = np.argmax(Y_pred_test, axis=1)
    Y_test_class = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_test_class == Y_test_class)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(Y_test_class,Y_pred_test_class)
    print('\nconfusion matrix, test samples:%d'%(X_test.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco:%s'%(i,rec))
