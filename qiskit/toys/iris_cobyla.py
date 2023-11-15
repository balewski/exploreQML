#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

'''

import numpy as np
from scipy.optimize import minimize
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import qiskit as qk
from pprint import pprint

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="exec on real backend, may take long time ")
    
    parser.add_argument('-b','--backName',default='ibmq_kolkata',help="backand for computations, should support dynamic circuits" )
    
    parser.add_argument('--ansatzReps',type=int,default=1, help="cycles in ansatz")
    parser.add_argument('-i','--maxIter',type=int,default=20, help="max COBYLA iterations")
    parser.add_argument('-n','--numShots',type=int,default=1000, help="shots")
    
    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
   
    return args

#...!...!....................
def get_iris_data():
    # Neural network parameters
    #input_size = 4

    #output_size = 3  # number of categories
    XmxIter=100  # for COBYLA

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Preprocess the data
    from sklearn.preprocessing import MinMaxScaler
    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    if 1:
        print('Iris: 1-hoy encoded labels')
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('data split:',X_train.shape, y_train.shape)
    return X_train, X_test, y_train, y_test 

#...!...!....................
def encode_features(X):
    from qiskit.circuit.library import ZZFeatureMap
    num_features = X.shape[1]
    feature_circ = ZZFeatureMap(feature_dimension=num_features, reps=1)
    return feature_circ

#...!...!....................
def myAnsatz(nFeat,nReps):
    from qiskit.circuit.library import EfficientSU2,TwoLocal
    ansatz_circ=EfficientSU2(num_qubits=nFeat, reps=nReps, entanglement='linear',skip_final_rotation_layer=True)
    # ansatz_circ = TwoLocal(num_qubits=nFeat, reps=nReps, rotation_blocks='ry', entanglement_blocks='cx', entanglement='circular')

    return ansatz_circ

#...!...!....................
def build_circuit(X,nReps):
    from qiskit import QuantumCircuit
    nFeat=X.shape[1]
    qc1=encode_features(X)
    print('\nFeatures'); print(qc1.decompose())
    qc2=myAnsatz(nFeat,nReps)
    print('\nAnsatz'); print(qc2.decompose())

    #nBits = len(qc2.clbits) # number of classical registers in the ansatz
    #nBits=nFeat
    nBits=num_label
    print('nBits=',nBits)
    circuit = QuantumCircuit(nFeat,nBits)
    circuit.append(qc1, range(nFeat))
    circuit.append(qc2, range(nFeat))
    circuit.measure(range(nBits),range(nBits))
    #circuit.measure(range(1,nBits+1),range(nBits))
    featNL=get_par_names(qc1,'features')
    ansatzNL=get_par_names(qc2,'ansatz')
 
    #circuit.measure(0,0)
    return circuit,featNL,ansatzNL

#...!...!....................
def get_par_names(qc,txt=''):
    parNL= [param.name for param in qc.parameters]
    print(txt+' parNL:',parNL)
    return parNL

#...!...!....................
def bind_features(qc,featName,X,nSamp=None):
    if nSamp==None: nSamp=X.shape[0]
    assert nSamp>0
    assert nSamp<=X.shape[0]
    # xD will map all input samples for all circuits
    xD=[ {} for i in range(nSamp) ]    
    for param in qc.parameters:
        if param.name not in featName: continue
        j=featName.index(param.name)
        for i in range(nSamp):
            xD[i][param]=X[i][j]
    
    assert len(xD[0])==len(featName) # all names must be found
    qcF=[ qc.assign_parameters(xD[i]) for i in range(nSamp) ]
    #print('assigned features to %d circuits'%nSamp)
    return qcF
   
#...!...!....................
def bind_weights(qcL,weightName,W):
    nCirc=len(qcL)
    qcW=[ None for i in range(nCirc) ]
    for ic in  range(nCirc):
        xD={}
        qc=qcL[ic]
        for param in qc.parameters:
            if param.name not in weightName: continue
            j=weightName.index(param.name)
            xD[param]=W[j]            
        assert len(xD)==len(weightName) # all names must be found
        qcW[ic]= qc.assign_parameters(xD)
    return qcW
   
        
# Function to extract a parameter by name
def get_parameter_by_name(circuit, name):
    for param in circuit.parameters:
        if param.name == name:
            return param
    return None

#...!...!....................
def M_test_full_bind_and_run():
    qcF=bind_features(qcT,featN,X_train,1)
    #print('features bound:');print(qcF[0])

    weights=init_weights(weightN)
    qcW=bind_weights(qcF,weightN,weights)
    print('weights bound:');print(qcW[0])
    
    # - - - -  FIRE JOB - - - - - - -
    job =  backend.run(qcW,shots=shots)
    jid=job.job_id()
    print('submitted JID=',jid,backend ,' now wait for execution of your circuit ...')
    
    result = job.result()

    counts = result.get_counts(0)
    print('counts:',backend); pprint(counts)
    if 0:
        labProb=HW_2_label(counts,mxLabel=num_label)
        print('HW labProb:',labProb)
    labProb=oneHot_2_label(counts,mxLabel=num_label)
    print('1hot labProb:',labProb)
    print('PASS')
    exit(0)

#...!...!....................
def HW_2_label(counts,mxLabel):
    hwV=np.zeros(mxLabel)
    totShots=0
    for bits,mshot in counts.items():
        hw=bits.count('1')
        #print(bits,hw,mshot)
        hwV[ hw%mxLabel]+=mshot
        totShots+=mshot
    #print('hwV:',hwV/totShots,'totShots:',totShots)
    return hwV/totShots

#...!...!....................
def oneHot_2_label(counts,mxLabel):
    hwV=np.zeros(mxLabel)
    totShots=0
    for bits,mshot in counts.items():
        xx=bits[:mxLabel]
        #print(bits,'xx:',xx,mshot)
        for k,v in enumerate(xx):
            if v=='1': hwV[ k]+=mshot       
        totShots+=mshot
        #break
    totSum=np.sum(hwV)
    #print('1hotV:',hwV,'totShots:',totShots,totSum)
    
    return hwV/totShots

#...!...!....................
def init_weights(weightN):
    nW=len(weightN)
    weights=np.random.rand(nW)
    #weights=np.zeros(nW)
    return weights

#...!...!....................
# Forward pass
def M_forward_pass(X, W):
    qcF=bind_features(qcT,featN,X)
    qcW=bind_weights(qcF,weightN,W)
    # - - - -  FIRE JOB - - - - - - -
    nCirc=len(qcW)
    job =  backend.run(qcW,shots=shots)
    jid=job.job_id()
    #print('submitted JID=',jid,backend ,nCirc,' circuits ...')
    
    result = job.result()    

    # compute density of labels for each circuit
    labDens=[ None for ic in range(nCirc)]
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        #labDens[ic]=oneHot_2_label(counts,mxLabel=num_label)
        labDens[ic]=HW_2_label(counts,mxLabel=num_label)
    return np.array(labDens)
    

#...!...!....................


#...!...!....................
def instantiate_circuit(qc,xName,pName,xV=None,pV=None):
    mx=len(xName)
    if xV==None: xV=np.random.rand(mx)

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nReps=args.ansatzReps
    
    X_train, X_test, y_train, y_test =get_iris_data()

    if 0: # reduce samples
        mxCirc=2
        X_train=X_train[:mxCirc]
        y_train=y_train[:mxCirc]

    num_feture=X_train.shape[1]
    num_label=3  # for Iris data
    
    qc,featN,weightN=build_circuit(X_train,nReps)
    print('M: full compact'); print(qc)
    
    backend = qk.Aer.get_backend('aer_simulator')
    qcT = qk.transpile(qc, backend=backend)
    print('M: transpiled');# print(qcT)

    shots=args.numShots
    #1M_test_full_bind_and_run()

    weights0=init_weights(weightN)
   
    # Loss history
    loss_history = []

    # Cross-entropy loss function
    def cross_entropy_loss(Y_pred, Y_true):
        #print('CEL',Y_pred.shape, Y_true.shape)
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m  # Adding a small value to prevent log(0)
        return loss


    # Loss function for optimization
    def loss_function(weights, X, Y):
        Y_pred_dens = M_forward_pass(X, weights)
        loss = cross_entropy_loss(Y_pred_dens, Y)
        #print('LF:',loss,Y)
        loss_history.append(loss)
        iIter=len(loss_history)
        if iIter%20==0: print('iter=%d loss=%.3f'%(iIter,loss))
        return loss

    #1loss_function(weights0, X_train, y_train)

    # Use COBYLA optimizer with set maximum of iterations

    result = minimize(fun=loss_function, 
                      x0=weights0, 
                      args=(X_train, y_train), 
                      method='COBYLA', 
                      options={'maxiter': args.maxIter})


    # Print loss history
    print("\nLoss History (Iteration vs. Loss):")
    for j, loss in enumerate(loss_history):
        if j%10==0 or j==args.maxIter-1:
            print(f"Iteration {j + 1}: Loss = {loss:.3f}")


    # Extract the optimized weights
    weightsOpt = result.x

    print('num par=',weightsOpt.shape)

    if 1: # Mustaffa's test
        # Test the model
        Y_pred_train = M_forward_pass(X_train,weightsOpt)
        Y_pred_train_class = np.argmax(Y_pred_train, axis=1)
        Y_train_class = np.argmax(y_train, axis=1)

        # Calculate accuracy
        accuracy = np.mean(Y_pred_train_class == Y_train_class)
        print(f"Train Accuracy: {accuracy * 100:.2f}%\n")
        #exit(0)
        
    # Test the model
    Y_pred_test = M_forward_pass(X_test,weightsOpt)
    Y_pred_test_class = np.argmax(Y_pred_test, axis=1)
    Y_test_class = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_test_class == Y_test_class)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(Y_test_class,Y_pred_test_class)
    print('\nconfusion matrix, test samples:%d'%(X_test.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco:%s'%(i,rec))
