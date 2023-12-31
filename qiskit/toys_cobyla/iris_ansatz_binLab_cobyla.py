#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
QML: 3-way classfier of Iris dataset
uses angle feature encoding (input)
uses binary labels  encoding (output)
Setup: classical minimizer + quantum-only ML layers
Quantum resources: 4 qubits connected linearily 

Example output  (has signifficant variability):

Pass1-best ends, 100 iterations, end-loss=-0.343
   bestIter=50, best loss=-0.375
num weights= (16,) weights/rad  in [-0.72,1.77],  avr=0.32 
Pass1-best test accuracy: 85.6%
confusion matrix, test samples: 90
true:0  reco: [28  2  0  5]
true:1  reco: [ 0 23  6  0]
true:2  reco: [ 0  0 26  0]
true:3  reco: [ 0  0  0  0]
'''

from pprint import pprint
import numpy as np
from scipy.optimize import minimize

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
    
import qiskit as qk
from qiskit.circuit import Parameter,ParameterVector

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')        
    parser.add_argument('--ansatzReps',type=int,default=3, help="cycles in ansatz")
    parser.add_argument('-i','--maxIter',type=int,default=100, help="max COBYLA iterations")
    parser.add_argument('-n','--numShots',type=int,default=2000, help="shots")
    parser.add_argument('-r','--rndSeed',type=int,default=42, help="seed for reproducibility of data split")
    parser.add_argument( "-D","--doublePass", action='store_true', default=False, help="executs 2nd COBYLA pass")
            
    args = parser.parse_args()
    args.cobyla_rhobeg=0.3 #  initial step size, smaller value prevents from too large rotations in Hilbert space
    args.input_scale=(0,np.pi) # for MinMaxScaler, must match myFeatureMap()
    args.test_fraction=0.6  # separates Iris samples, small training size is used to save shots on HW
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args


#...!...!....................
# Function to encode integer labels shape (nSamp,) to be (nSamp,MB) to match reading nBit on QPU
def M_binary_encode_labels(labels):
    nBit=int(np.ceil(np.log2(num_label)))
    nSamp=labels.shape[0]
    MB=1<<nBit  # number of possible bitstrings
    # reward and penalyze schem:
    lab_bin= np.full((nSamp,MB),-1./MB) # blank penalty for any bistring
    # Set the corresponding indices  value to 1   : this is the reward for good label
    np.put_along_axis(lab_bin, labels[:, None], 1, axis=1)
    return lab_bin

#...!...!....................
def get_iris_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    num_label=3  # for Iris data
    
    # Preprocess the data
    # Normalize data
    scaler = MinMaxScaler(feature_range=args.input_scale)
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_fraction, random_state=args.rndSeed)
    
    print('Iris data split  train:',X_train.shape, ', test:',X_test.shape)
    return X_train, X_test, y_train, y_test ,num_label

#...!...!....................
def myFeatureMap(X):
    nFeat=X.shape[1]
    qc = qk.QuantumCircuit(nFeat,name='angleMap')
    alphas = ParameterVector('alpha',length=nFeat)
    
    for i in range(nFeat):
        qc.ry( alphas[i],i)
    qc.barrier()
    return qc

#...!...!....................
def rotation_block(circ, ir, addRz=True,nq=None):
    # assumes nb of params scales with  num of qubits
    if nq==None: nq=circ.num_qubits
    nPar=2*nq-1 if addRz else nq
    thetas = ParameterVector('th%d'%ir, length=nPar)
    for i in range(nq): circ.ry(thetas[i],i)
    if addRz:
        for i in range(nq-1):  circ.rz(thetas[i+nq],i)

#...!...!....................
def entangling_block(qc,cxDir=0,doCircle=False):
    nq=qc.num_qubits
    for i in range(nq-1):
        ictr=i; itgt=ictr+1
        if cxDir:  ictr,itgt=itgt,ictr
        qc.cx(ictr,itgt)
    if doCircle: qc.cx(nq-1,0)

#...!...!....................
def myAnsatz(nFeat,nReps, barrier=True, final_rotation=True):
    qc = qk.QuantumCircuit(nFeat,name='ansatz_r%d'%nReps)
    for ir in range(nReps):
        rotation_block(qc,ir, addRz=False)
        if barrier: qc.barrier()
        entangling_block(qc,cxDir=0,doCircle=False)
        if barrier: qc.barrier()
    if final_rotation==True:
        rotation_block(qc,nReps, addRz=False)
    return qc
    
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
    weights=np.random.rand(nW)*0.1  
    #weights=np.zeros(nW)
    return weights

#...!...!....................
def M_loss_function(weights, X, Y):
    Y_pred_dens = M_forward_pass(X, weights)
    loss = binary_cross_entropy_loss(Y_pred_dens, Y)
    myMem.add(loss)
    return loss

#...!...!....................
def bin_label_decoder(counts,mxLabel):    
    probV=np.zeros(mxLabel)
    totShots=0    
    for bits,mshot in counts.items():
        # Convert the bit string to an integer
        mval = int(bits, 2)
        probV[ mval]=mshot
        totShots+=mshot
    probV/=totShots
    #print(counts);print('probV:',probV,'totShots:',totShots)
    return probV


#...!...!....................
def M_forward_pass(X, W):
    global weightCurrent
    #print('MFP:',X.shape,W.shape)
    qcF=bind_features(qcT,featN,X)
    qcW=bind_weights(qcF,weightN,W)    
    nCirc=len(qcW)
    
    # - - - -  FIRE Qiskit JOB - - - - - - -
    job =  backend.run(qcW,shots=args.numShots)
    jid=job.job_id()
    #print('submitted JID=',jid,backend ,nCirc,' circuits ...')    
    result = job.result()    
    weightCurrent=W.copy()
    
    # compute probability density of measured labels for each circuit
    labDens=[ None for ic in range(nCirc)] # lock memory
    assert num_label<=4  # for mxLabels, tmp
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        labDens[ic]=bin_label_decoder(counts,mxLabel=4)
    return np.array(labDens)
    

#...!...!....................
def binary_cross_entropy_loss(Y_pred, Y_true):    
    #print('BCEL:',Y_pred.shape, Y_true.shape)
    # Adding a small value to prevent log(0)    
    loss= -np.mean(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred + 1e-9))
    return loss

#...!...!....................
def M_evaluate( weightsOpt,txt=''):
    if 0:    # Print loss history
        print("\n%s loss History (Iteration vs. Loss):"%txt)
        for j, loss in enumerate(loss_hist):
            if j%10==0 or j==args.maxIter-1:
                print(f"Iteration {j + 1}: Loss = {loss:.3f}")

    # Extract the optimized weights
    wMin=min(weightsOpt)
    wMax=max(weightsOpt)
    wAvr=np.mean(weightsOpt)

    myMem.report(txt)
    print('num weights=',weightsOpt.shape,'weights/rad  in [%.2f,%.2f],  avr=%.2f '%(wMin,wMax,wAvr))
    
    # Test the model
    y_pred_dens = M_forward_pass(X_test,weightsOpt)
    y_pred_test = np.argmax(y_pred_dens, axis=1) # do: majority wins

    # Calculate accuracy
    accuracy = np.mean(y_pred_test == y_test)
    print("%s test accuracy: %.1f%c"%( txt,accuracy *100,37 ))

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test,y_pred_test)
    print('confusion matrix, test samples: %d'%(X_test.shape[0]))
    for i,rec  in enumerate(conf_matrix):
        print('true:%d  reco: %s'%(i,rec))
    print()
    


#...!...!....................
def build_circuit(X,nReps):
    nFeat=X.shape[1]
    nBits=2; assert 3==num_label
  
    qc1=myFeatureMap(X)
    print('\nFeatures'); print(qc1)
    qc2=myAnsatz(nFeat,nReps, final_rotation=True)
    print('\nAnsatz'); print(qc2)

    print('nBits=',nBits)
    circuit = qk.QuantumCircuit(nFeat,nBits)
    circuit.append(qc1, range(nFeat))
    circuit.append(qc2, range(nFeat))
    moff=2  # decide which qubits to read, 0 is the worse choice ?!?
    circuit.measure(range(moff,moff+nBits),range(nBits))
    
    featNL=get_par_names(qc1,'features')
    ansatzNL=get_par_names(qc2,'ansatz')
 
    return circuit,featNL,ansatzNL


#...!...!....................
class trainMemory():  # book keeping class
    def __init__(self):
        self.loss=[]      # Loss history
        self.bestIter=-1
        self.bestLoss=1e99
        
    def add(self,val):
        global weightCurrent
        self.loss.append(val)
        iIter=self.num_iter()
        if self.bestLoss> val :
            self.bestLoss=val
            self.bestIter=iIter
            self.bestW=weightCurrent
        if iIter%5==0: print('iter=%d loss=%.3f'%(iIter,val))
        
    def report(self,txt):
        lastIter=self.num_iter()
        print('\n%s ends, %d iterations, end-loss=%.3f'%(txt,lastIter,self.loss[-1]))
        if lastIter!=self.bestIter:
            print('   bestIter=%d, best loss=%.3f'%(self.bestIter,self.bestLoss))
        else:
            self.bestW=weightCurrent
            
    def num_iter(self): return  len(self.loss)
#...!...!....................        
        

#=================================
#=================================
#  M A I N
#=================================
#=================================
weightCurrent=None  # global variable hack, tmp
if __name__ == "__main__":
    args=get_parser()
     
    X_train, X_test, y_train, y_test, num_label =get_iris_data()
    y_train_bin=M_binary_encode_labels(y_train)
    
    num_feture=X_train.shape[1]
   
    qc,featN,weightN=build_circuit(X_train,args.ansatzReps)
    num_weight=len(weightN)
    print('M: full compact, num_weight=%d'%num_weight); print(qc)
    
    
    backend = qk.Aer.get_backend('aer_simulator')
    qcT = qk.transpile(qc, backend=backend)
    print('M: transpiled');# print(qcT)

    weightsIni=init_weights(weightN) 
    myMem=trainMemory()

    #1M_forward_pass( X_train,weightsIni)
    #1M_loss_function(weightsIni, X_train, y_train_bin)    
 
    # Use COBYLA optimizer with set maximum of iterations
    result = minimize(fun=M_loss_function, 
                      x0=weightsIni, 
                      args=(X_train, y_train_bin), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter,'rhobeg': args.cobyla_rhobeg})
    #'rhobeg' controls the initial step size of the parameters when computing the next value of the function, default is 1.0. Smaller step increases number of iterations

    weightsOpt = result.x
    M_evaluate(weightsOpt,txt='Pass1-end')
    M_evaluate(myMem.bestW,txt='Pass1-best')
    
    
    if not args.doublePass :  exit(0)

    # =============================================
    rhobeg= args.cobyla_rhobeg*2
    print('2nd COBYLA, rhobeg=%.2f ...'%rhobeg)

    # Reset loss history
    myMem=trainMemory()
    result = minimize(fun=M_loss_function, 
                      x0=weightsOpt, 
                      args=(X_train, y_train_bin), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter//2,'rhobeg': rhobeg})

    M_evaluate(weightsOpt,txt='Pass2-end')
    M_evaluate(myMem.bestW,txt='Pass2-best')
 
   
