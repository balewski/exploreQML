#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
use Qiskit Ansatz

Example output  (has signifficant variability)

Iteration 181: Loss = 0.842
num weights= (16,) weights/rad  in [-1.64,1.29],  avr=0.32 

Pass1 test accuracy: 63.3%
confusion matrix, test samples: 30
true:0  reco:[7 1 2]
true:1  reco:[0 8 1]
true:2  reco:[1 6 4]

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
    parser.add_argument('--ansatzReps',type=int,default=2, help="cycles in ansatz")
    parser.add_argument('-i','--maxIter',type=int,default=300, help="max COBYLA iterations")
    parser.add_argument('-n','--numShots',type=int,default=2000, help="shots")
    parser.add_argument( "-D","--doublePass", action='store_true', default=False, help="executs 2nd COBYLA pass")
            
    args = parser.parse_args()
    args.rnd_seed=42       # for reproducibility of data split
    args.cobyla_rhobeg=0.3 #  initial step size
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args

#...!...!....................
def get_iris_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    num_label=3  # for Iris data
    
    # Preprocess the data
    from sklearn.preprocessing import MinMaxScaler
    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    print('Iris: 1-hoy encoded labels')
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.rnd_seed)
    
    print('data split:',X_train.shape, y_train.shape)
    return X_train, X_test, y_train, y_test ,num_label

#...!...!....................
def encode_features(X):
    from qiskit.circuit.library import ZZFeatureMap
    num_features = X.shape[1]
    feature_circ = ZZFeatureMap(feature_dimension=num_features, reps=1)
    return feature_circ

#...!...!....................
def myAnsatzQiskit(nFeat,nReps):
    from qiskit.circuit.library import EfficientSU2,TwoLocal
    ansatz_circ=EfficientSU2(num_qubits=nFeat, reps=nReps, entanglement='linear',skip_final_rotation_layer=True)
    return ansatz_circ

#...!...!....................
def get_par_names(qc,txt=''):
    parNL= [param.name for param in qc.parameters]
    print(txt+' %d parNL:'%len(parNL),parNL)
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
   
#...!...!....................
def init_weights(weightN):
    nW=len(weightN)
    weights=np.random.rand(nW)*0.5  
    #weights=np.zeros(nW)
    return weights

#...!...!....................
def M_loss_function(weights, X, Y):
    Y_pred_dens = M_forward_pass(X, weights)
    loss = cross_entropy_loss(Y_pred_dens, Y)
    loss_history.append(loss)
    iIter=len(loss_history)
    if iIter%5==0: print('iter=%d loss=%.3f'%(iIter,loss))
    return loss

#...!...!....................
def HW_2_label(counts,mxLabel):
    hwV=np.zeros(mxLabel)
    totShots=0
    for bits,mshot in counts.items():
        hw=bits.count('1')
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
        for k,v in enumerate(xx):
            if v=='1': hwV[ k]+=mshot       
        totShots+=mshot
    totSum=np.sum(hwV)
    #print('1hotV:',hwV,'totShots:',totShots,totSum)    
    return hwV/totShots

#...!...!....................
def M_forward_pass(X, W):
    qcF=bind_features(qcT,featN,X)
    qcW=bind_weights(qcF,weightN,W)    
    nCirc=len(qcW)
    
    # - - - -  FIRE JOB - - - - - - -
    job =  backend.run(qcW,shots=args.numShots)
    jid=job.job_id()
    #print('submitted JID=',jid,backend ,nCirc,' circuits ...')    
    result = job.result()    

    # compute density of labels for each circuit
    labDens=[ None for ic in range(nCirc)]
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        # pick label extraction scheme here:
        #labDens[ic]=oneHot_2_label(counts,mxLabel=num_label)
        labDens[ic]=HW_2_label(counts,mxLabel=num_label)
    return np.array(labDens)
    
#...!...!....................
# Cross-entropy loss function
def cross_entropy_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m  # Adding a small value to prevent log(0)
    return loss

#...!...!....................
def M_evaluate(txt=''):
    # Print loss history
    print("\n%s loss History (Iteration vs. Loss):"%txt)
    for j, loss in enumerate(loss_history):
        if j%10==0 or j==args.maxIter-1:
            print(f"Iteration {j + 1}: Loss = {loss:.3f}")

    # Extract the optimized weights
    weightsOpt = result.x
    wMin=min(weightsOpt)
    wMax=max(weightsOpt)
    wAvr=np.mean(weightsOpt)
    
    print('num weights=',weightsOpt.shape,'weights/rad  in [%.2f,%.2f],  avr=%.2f '%(wMin,wMax,wAvr))

    # Test the model
    Y_pred_test = M_forward_pass(X_test,weightsOpt)
    Y_pred_test_class = np.argmax(Y_pred_test, axis=1)
    Y_test_class = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_test_class == Y_test_class)
    print("\n%s test accuracy: %.1f%c"%( txt,accuracy *100,37 ))

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(Y_test_class,Y_pred_test_class)
    print('confusion matrix, test samples: %d'%(X_test.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco:%s'%(i,rec))
    print()
    return weightsOpt


#...!...!....................
def build_circuit(X,nReps):
    from qiskit import QuantumCircuit
    nFeat=X.shape[1]
    qc1=encode_features(X)
    print('\nFeatures'); print(qc1.decompose())
    qc2=myAnsatzQiskit(nFeat,nReps)
    print('\nAnsatz'); print(qc2.decompose())

    nBits=num_label
    print('nBits=',nBits)
    circuit = QuantumCircuit(nFeat,nBits)
    circuit.append(qc1, range(nFeat))
    circuit.append(qc2, range(nFeat))
    circuit.measure(range(nBits),range(nBits))
    
    featNL=get_par_names(qc1,'features')
    ansatzNL=get_par_names(qc2,'ansatz')
 
    return circuit,featNL,ansatzNL

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
     
    X_train, X_test, y_train, y_test,num_label =get_iris_data()
    num_feture=X_train.shape[1]    

    qc,featN,weightN=build_circuit(X_train,args.ansatzReps)
    num_weight=len(weightN)
    print('M: full compact, num_weight=%d'%num_weight); print(qc)
    
    backend = qk.Aer.get_backend('aer_simulator')
    qcT = qk.transpile(qc, backend=backend)
    print('M: transpiled');# print(qcT)

    weightsIni=init_weights(weightN)
    #1loss_function(weights0, X_train, y_train)

    # Loss history
    loss_history = []

    # Use COBYLA optimizer with set maximum of iterations
    result = minimize(fun=M_loss_function, 
                      x0=weightsIni, 
                      args=(X_train, y_train), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter,'rhobeg': args.cobyla_rhobeg})
    #'rhobeg' controls the initial step size of the parameters when computing the next value of the function, default is 1.0. Smaller step increases number of iterations

   
    weightsOpt=M_evaluate(txt='Pass1')

    if not args.doublePass :  exit(0)
    
    # =============================================
    rhobeg= args.cobyla_rhobeg*2
    print('2nd COBYLA, rhobeg=%.2f ...'%rhobeg)

    # Reset loss history
    loss_history = []
    result = minimize(fun=loss_function, 
                      x0=weightsOpt, 
                      args=(X_train, y_train), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter//2,'rhobeg': rhobeg})

    weightsOpt=M_evaluate(txt='Pass2')
   
