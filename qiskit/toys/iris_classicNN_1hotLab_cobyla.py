#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Jan:
give my python code using only numpy and COBYLA which reads in iris data set, constructs ML classifier consisting of 3 layers of fully connected network  4-5-3. The last layer should represent the 3 classes of iris hot encoded. COBYLA should be used as minimizer
Use cross-entropy loss
add bias terms to fully connected layers
Print loss vs. iteration at the end

Code below is moslty generated by ChatGPT4
'''

'''
Iteration 800: Loss = 0.060
num par= (19,) W in [-4.68,6.71] avr=0.69
Accuracy: 100.00%

confusion matrix, test samples:30
true:0  reco:[10  0  0]
true:1  reco: [0  9  0]
true:2  reco:[ 0  0 11]


*** 50 iter, xnetr=0.62 , acc=70%
confusion matrix, test samples:30
true:0  reco:[10  0  0]
true:1  reco:[0 0 9]
true:2  reco:[ 0  0 11]

*** 100 iter, Accuracy: 90.00%

confusion matrix, test samples:30
true:0  reco:[10  0  0]
true:1  reco:[0 6 3]
true:2  reco:[ 0  0 11]

*** 200 iter, xnetr=0.34, Accuracy: 93.33%
confusion matrix, test samples:30
true:0  reco:[10  0  0]
true:1  reco:[0 7 2]
true:2  reco:[ 0  0 11]
'''

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural network parameters
input_size = 4
hidden_size = 1
output_size = 3  # number of categories
mxIter=800  # for COBYLA
rnd_seed=42 # for reproducibility of data split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=rnd_seed)


# Initialize weights
np.random.seed(rnd_seed)
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Forward pass including biases
def forward_pass(X, params):
    # Unpack parameters
    W1 = params[:input_size * hidden_size].reshape(input_size, hidden_size)
    b1 = params[input_size * hidden_size:input_size * hidden_size + hidden_size]
    W2 = params[input_size * hidden_size + hidden_size:input_size * hidden_size + hidden_size + hidden_size * output_size].reshape(hidden_size, output_size)
    b2 = params[input_size * hidden_size + hidden_size + hidden_size * output_size:]

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2


# Cross-entropy loss function
def cross_entropy_loss(Y_pred, Y_true):
    #print('CNE',Y_pred.shape, Y_true.shape); one_hot # CNE (120, 3) (120, 3)
    m = Y_true.shape[0]
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m  # Adding a small value to prevent log(0)
    return loss

# Loss function for optimization including biases
def loss_function(params, X, Y):    
    Y_pred = forward_pass(X, params)
    loss = cross_entropy_loss(Y_pred, Y)
    loss_history.append(loss)
    return loss

# Flatten weights and biases for optimization
initial_weights = np.hstack((W1.ravel(), b1, W2.ravel(), b2))


# Loss history
loss_history = []

if 1: # Use COBYLA optimizer with set maximum of iterations
    result = minimize(fun=loss_function, 
                  x0=initial_weights, 
                  args=(X_train, y_train), 
                  method='COBYLA', 
                  options={'maxiter': mxIter,'rhobeg': 0.3})
    # 'rhobeg' controls the initial step size of the parameters when computing the next value of the function, default is 1.0. Smaller step increases number of iterations

if 0: # Use L-BFGS-B optimizer with bounds
    bounds = [(-3, 3)] * len(initial_weights)
    result = minimize(fun=loss_function, 
                  x0=initial_weights, 
                  args=(X_train, y_train), 
                  method='L-BFGS-B', 
                  bounds=bounds, 
                  options={'maxfun': mxIter})

if 0: # Basin-Hopping optimization
    bounds = [(-np.pi, np.pi)] * len(initial_weights)
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,
                        "args": (X_train, y_train)}
    result = basinhopping(func=loss_function,
                          x0=initial_weights,
                          minimizer_kwargs=minimizer_kwargs,
                          niter=mxIter
                          )

if 0: # Use differential_evolution for optimization
    bounds = [(-np.pi, np.pi)] * len(initial_weights)
    result = differential_evolution(func=loss_function, 
                                bounds=bounds, 
                                args=(X_train, y_train), 
                                maxiter=mxIter, # number of generations
                                popsize=15,   # Adjusted population size
                                strategy='best1bin')

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
