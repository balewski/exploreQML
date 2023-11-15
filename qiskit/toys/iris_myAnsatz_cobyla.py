#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
use my custom Ansatz
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
    
    parser.add_argument('--ansatzReps',type=int,default=2, help="cycles in ansatz")
    parser.add_argument('-i','--maxIter',type=int,default=300, help="max COBYLA iterations")
    parser.add_argument('-n','--numShots',type=int,default=2000, help="shots")
    parser.add_argument( "-D","--doublePass", action='store_true', default=False, help="executs 2nd COBYLA pass")
     
    args = parser.parse_args()

    args.rnd_seed=42       # for reproducibility of data split
    args.cobyla_rhobeg=0.3 #  initial step size
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
   
    return args

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter,ParameterVector

def rotation_block(circ, ir, addRz=True,nq=None):
    # assumes nb of params scales with  num of qubits
    if nq==None:
        nq=circ.num_qubits
    nPar=2*nq-1 if addRz else nq
    thetas = ParameterVector('th%d'%ir, length=nPar)
    for i in range(nq): circ.ry(thetas[i],i)
    if addRz:    
        for i in range(nq-1):  circ.rz(thetas[i+nq],i)

def entangling_block2_2(circ):
    nq=circ.num_qubits
    n2=nq//2
    for i in range(n2): 
        ict=2*i; itg=ict+1
        circ.cx(ict,itg)
    for i in range(n2): 
        ict=2*i+1; itg=(ict+1)%nq
        circ.cx(ict,itg)

def entangling_block_ladder(circ):
    nq=circ.num_qubits
    for i in range(nq-1):  circ.cx(i,i+1)

                             
def non_linear1(circuit,iq,ir):
    beta = Parameter('be%d_%d'%(ir,iq))
    #beta=np.pi/2 # for midMeas
    # the circuit, the parameter and the qubit of interest
    circuit.ry(beta, iq)
    circuit.measure(iq, 0)
    
    
def non_linear(qc,iq,ir):
    beta = Parameter('be%d_%d'%(ir,iq))    
    qc.ry(beta, iq)
    qc.reset(iq)
    qc.h(iq)
    qc.cx(iq,0)
        

def myAnsatz(nFeat, reps, nMeas, midMeasLL, barrier=True, final_rotation=True):
    assert len(midMeasLL)==reps
    qubits = QuantumRegister(nFeat,'q')
    meas = ClassicalRegister(nMeas,'c')
    qc = QuantumCircuit(qubits, meas)

    for ir in range(reps):
        rotation_block(qc,ir)
        if barrier: qc.barrier()
        entangling_block_ladder(qc)
        if barrier: qc.barrier()
        midmL=midMeasLL[ir]
        for iq in midmL: non_linear(qc,iq,ir)            
        if barrier: qc.barrier()
    
    #final rotation block, optional
    if final_rotation:
       rotation_block(qc,ir+1,addRz=False, nq=nFeat//2)
    if barrier: qc.barrier()
    #qc.measure(range(nMeas),meas)
 
    return qc
''' Use case
# Anzatz circuit
nReps=3
midMeasLL=[ [] for i in range(nReps) ] # no midMeas at all
midMeasLL=[[2],[3],[]]  # select midMeas qubits per repetition
ansatz_circ = myAnsatz(num_features, reps=nReps, nMeas=2, midMeasLL=midMeasLL, barrier=True, final_rotation=False)
'''

#...!...!....................
def get_iris_data():
    num_label=3  # for Iris data
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
def myAnsatzJan(nFeat,nReps):
    midMeasLL=[ [] for i in range(nReps) ] # no midMeas at all
    #midMeasLL=[[0,1],[2,3]]  # select qubits for non-linearity, per repetition
    #midMeasLL=[[3],[]]  
    ansatz_circ = myAnsatz(nFeat, reps=nReps, nMeas=num_label, midMeasLL=midMeasLL, barrier=True, final_rotation=False)
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
    job =  backend.run(qcW,shots=args.numShots)
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
def init_weights(weightN):
    nW=len(weightN)
    weights=np.random.rand(nW)*0.5  
    #weights=np.zeros(nW)
    return weights

#...!...!....................
# Loss function for optimization
def M_loss_function(weights, X, Y):
    Y_pred_dens = M_forward_pass(X, weights)
    loss = cross_entropy_loss(Y_pred_dens, Y)
    #print('LF:',loss,Y)
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
    totSum=np.sum(hwV)
    #print('1hotV:',hwV,'totShots:',totShots,totSum)    
    return hwV/totShots

#...!...!....................
# Forward pass
def M_forward_pass(X, W):
    qcF=bind_features(qcT,featN,X)
    qcW=bind_weights(qcF,weightN,W)
    # - - - -  FIRE JOB - - - - - - -
    nCirc=len(qcW)
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
    #print('CEL',Y_pred.shape, Y_true.shape)
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
    
    print('num par=',weightsOpt.shape,'weights/rad  in [%.2f,%.2f],  avr=%.2f '%(wMin,wMax,wAvr))

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
    print('\nconfusion matrix, test samples:%d'%(X_test.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco:%s'%(i,rec))

    return weightsOpt

#...!...!....................
def build_circuit(X,nReps):
    from qiskit import QuantumCircuit
    nFeat=X.shape[1]
    qc1=encode_features(X)
    print('\nFeatures'); print(qc1.decompose())
    #1qc2=myAnsatzQiskit(nFeat,nReps)
    qc2=myAnsatzJan(nFeat,nReps)
    print('\nAnsatz'); print(qc2.decompose())

    nBits=num_label
    print('nBits=',nBits)
    circuit = QuantumCircuit(nFeat,nBits)
    circuit.append(qc1, range(nFeat))
    circuit.append(qc2, range(nFeat),range(nBits))
    circuit.measure(range(nBits),range(nBits))
    
    featNL=get_par_names(qc1,'features')
    ansatzNL=get_par_names(qc2,'ansatz')
 
    #circuit.measure(0,0)
    return circuit,featNL,ansatzNL

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nReps=args.ansatzReps
    
    X_train, X_test, y_train, y_test, num_label =get_iris_data()
    num_feture=X_train.shape[1]
 
    if 0: # reduce samples
        mxCirc=2
        X_train=X_train[:mxCirc]
        y_train=y_train[:mxCirc]
   
    
    qc,featN,weightN=build_circuit(X_train,nReps)
    num_weight=len(weightN)
    print('M: full compact, num_weight=%d'%num_weight); print(qc)
    
    backend = qk.Aer.get_backend('aer_simulator')
    qcT = qk.transpile(qc, backend=backend)
    print('M: transpiled');# print(qcT)

    #1M_test_full_bind_and_run()

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
    
    if 1: # Mustaffa's test
        # Test the model
        Y_pred_train = M_forward_pass(X_train,weightsOpt)
        Y_pred_train_class = np.argmax(Y_pred_train, axis=1)
        Y_train_class = np.argmax(y_train, axis=1)

        # Calculate accuracy
        accuracy = np.mean(Y_pred_train_class == Y_train_class)
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        #exit(0)
        
    if not args.doublePass :  exit(0)
    
    # =============================================
    rhobeg= args.cobyla_rhobeg*2
    print('2nd COBYLA, rhobeg=%.2f ...'%rhobeg)

    # Loss history
    loss_history = []

    # Use COBYLA optimizer with set maximum of iterations
    result = minimize(fun=loss_function, 
                      x0=weightsOpt, 
                      args=(X_train, y_train), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter//2,'rhobeg': rhobeg})

    weightsOpt=M_evaluate(txt='Pass2')
   
