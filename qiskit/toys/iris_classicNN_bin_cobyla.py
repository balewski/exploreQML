#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Jan:
give my python code using only numpy and COBYLA which reads in iris data set, constructs ML classifier consisting of 3 layers of fully connected network  4-5-3. The last layer should represent the 3 classes of iris binary.

 COBYLA should be used as minimizer
Use cross-entropy loss
add bias terms to fully connected layers
Print loss vs. iteration at the end

Please change the logic. make the last layer to have 2 cells. Use sigmoid activation for the last layer.  Map 2 output floats as 1 bits. Interprte thos 2 bits as binary encoding of the label. Construct loss approariatly to match this binary encoding

Code below is moslty generated by ChatGPT4
'''

'''
Output:

Iteration 1500: Loss = 0.054
num par= (37,) W in [-7.76,7.59] avr=-0.45
Accuracy: 100.00%

confusion matrix, test samples:30
true:0  reco:[10  0  0]
true:1  reco:[ 0  9  0]
true:2  reco:[ 0  0 11]

'''
import numpy as np
from scipy.optimize import minimize
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Neural network parameters
input_size = 4
hidden_size = 5
output_size = 2  # Two output neurons for binary encodin
mxIter=1500 # for COBYLA
rnd_seed=42 # for reproducibility of data split

# Function to encode labels as 2-bit binary
def binary_encode_labels(labels):
    # Binary encoding: 00, 01, 10
    binary_encoded = np.array([[0, 0], [0, 1], [1, 0]])[labels]
    return binary_encoded

# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Binary encode the labels
y_binary_encoded = binary_encode_labels(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary_encoded, test_size=0.2, random_state=rnd_seed)


# Activation functions and forward pass
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(X, params):
    W1 = params[:end_W1].reshape(input_size, hidden_size)
    b1 = params[end_W1:end_b1]
    W2 = params[end_b1:end_W2].reshape(hidden_size, output_size)
    b2 = params[end_W2:]

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # Sigmoid output for binary encoding
    return A2

# Binary cross-entropy loss function
def binary_cross_entropy_loss(Y_pred, Y_true):
    return -np.mean(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred + 1e-9))

# Initialize weights and biases
#1np.random.seed(rnd_seed)
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)

# Flatten weights and biases for optimization
initial_params = np.hstack((W1.ravel(), b1, W2.ravel(), b2))

# Define the end indices for each set of parameters
end_W1 = input_size * hidden_size
end_b1 = end_W1 + hidden_size
end_W2 = end_b1 + hidden_size * output_size

# Loss function including biases
def loss_function(params, X, Y):
 
    Y_pred = forward_pass(X, params)
    loss=binary_cross_entropy_loss(Y_pred, Y)
    loss_history.append(loss)
    return loss

# Loss history
loss_history = []

# Use COBYLA optimizer
result = minimize(fun=loss_function, 
                  x0=initial_params, 
                  args=(X_train, y_train), 
                  method='COBYLA',
                  options={'maxiter': mxIter,'rhobeg': 0.8})

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
Y_pred_test_binary = (Y_pred_test > 0.5).astype(int)

# Convert binary predictions back to class labels
def binary_to_label(binary_array):
    return np.array([2 if row[0] == 1 else row[1] for row in binary_array])

Y_pred_test_labels = binary_to_label(Y_pred_test_binary)
y_test_labels = binary_to_label(y_test)

# Calculate accuracy
accuracy = np.mean(Y_pred_test_labels == y_test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Compute the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test_labels,Y_pred_test_labels)
print('\nconfusion matrix, test samples:%d'%(X_test.shape[0]))
for i,rec  in enumerate(conf_matrix):
    print('true:%d  reco:%s'%(i,rec))
