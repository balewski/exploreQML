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

import os
from pprint import pprint
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
    
import qiskit as qk
from qiskit.circuit import Parameter,ParameterVector
from Util_iris_binLab import get_iris_data, binary_cross_entropy_loss, init_weights, get_par_names, bind_features, bind_weights , TrainHistory


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')        
    parser.add_argument('--ansatzReps',type=int,default=3, help="cycles in ansatz")
    parser.add_argument('-i','--maxIter',type=int,default=300, help="max COBYLA iterations")
    parser.add_argument('-n','--numShots',type=int,default=2000, help="shots")
    parser.add_argument('-r','--rndSeed',type=int,default=42, help="seed for reproducibility of data split")
    parser.add_argument( "-D","--doublePass", action='store_true', default=False, help="executs 2nd COBYLA pass")
    parser.add_argument('--expName',default='test123', help="name of experiment")
    parser.add_argument('--myRank',type=int,default=0, help="indexing for parallel execution")
            
    args = parser.parse_args()
    args.cobyla_rhobeg=0.3 #  initial step size, smaller value prevents from too large rotations in Hilbert space
    args.input_scale=(0,np.pi) # for MinMaxScaler, must match myFeatureMap()
    args.train_fraction=0.7  # separates Iris samples, small training size is used to save shots on HW
    if args.myRank>0:
        args.verb=0
        args.rndSeed+=args.myRank
    else:
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args


#...!...!....................
# Function to encode integer labels shape (nSamp,) to be (nSamp,MB) to match reading nBit on QPU
def M_binary_encode_labels(labels):
    nBit=int(np.ceil(np.log2(num_label)))
    nSamp=labels.shape[0]
    MB=1<<nBit  # number of possible bitstrings
    # reward and penalyze schem:
    #Xlab_bin= np.full((nSamp,MB),0) # blank penalty for any bistring
    lab_bin= np.zeros((nSamp,MB))
    # Set the corresponding indices  value to 1   : this is the reward for good label
    np.put_along_axis(lab_bin, labels[:, None], 1, axis=1)
    return lab_bin

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
def nonlin_block(qc):
    nq=qc.num_qubits
    qc.reset(nq-1)
   
#...!...!....................
def add_meas_block(qc,ir):
    nQ=qc.num_qubits
    nB=qc.num_clbits
    moff=2
    thetas = ParameterVector('th%d'%ir, length=nB)
    for i in range(nB):
        iq=i+moff
        qc.ry(thetas[i],iq)
        qc.measure(iq,i)
       

#...!...!....................
def myAnsatz(nFeat,nReps, meas_list,barrier=True):
    nBits=len(meas_list)
    assert nBits>0
    qc = qk.QuantumCircuit(nFeat,nBits,name='ansatz_r%d'%nReps)
    for ir in range(nReps):
        rotation_block(qc,ir, addRz=False)
        if barrier: qc.barrier()
        entangling_block(qc,cxDir=0,doCircle=False)
        #if ir==0: nonlin_block(qc)
        if barrier: qc.barrier()
    add_meas_block(qc,ir=nReps)
    return qc

#...!...!....................
def M_loss_function(weights, X, Y):
    Y_pred_dens = M_forward_pass(X, weights)
    loss = binary_cross_entropy_loss(Y_pred_dens, Y)
    myHist.add(loss)
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
    global myHist    #print('MFP:',X.shape,W.shape)
    qcF=bind_features(qcT,featN,X)
    qcW=bind_weights(qcF,weightN,W)    
    nCirc=len(qcW)
    
    # - - - -  FIRE Qiskit JOB - - - - - - -
    job =  backend.run(qcW,shots=args.numShots)
    jid=job.job_id()
    #print('submitted JID=',jid,backend ,nCirc,' circuits ...')    
    result = job.result()    
    myHist.weightCurrent=W.copy()
    
    # compute probability density of measured labels for each circuit
    labDens=[ None for ic in range(nCirc)] # lock memory
    assert num_label<=4  # for mxLabels, tmp
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        labDens[ic]=bin_label_decoder(counts,mxLabel=4)
    return np.array(labDens)
    

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

    myHist.report(txt)
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
    return accuracy

#...!...!....................
def summary_line(acc):
    iIter=myHist.num_iter()
    loss=myHist.loss[-1]
    elaT=myHist.elaTime
    
    print('#sum,%s,%.2f,%d,%.3f,%d,%.1f'%(args.expName,acc,iIter,loss,args.myRank,elaT))

#...!...!....................
def build_circuit(X,nReps,verb=1):
    nFeat=X.shape[1]
    nBits=2; assert 3==num_label
  
    qc1=myFeatureMap(X)
    if verb>0: print('\nFeatures'); print(qc1)
    measL=[2,3]
    qc2=myAnsatz(nFeat,nReps, measL)
    if verb>0: print('\nAnsatz'); print(qc2)

    #print('nBits=',nBits)
    
    circuit = qk.QuantumCircuit(nFeat,nBits)
    circuit.append(qc1, range(nFeat))
    circuit.append(qc2, range(nFeat),range(nBits)) 
     
    featNL=get_par_names(qc1,'features',verb)
    ansatzNL=get_par_names(qc2,'ansatz',verb)#+get_par_names(qc3,'meas',verb)
 
    return circuit,featNL,ansatzNL

#...!...!....................                

#=================================
#=================================
#  M A I N
#=================================
#=================================
#weightCurrent=None  # global variable hack, tmp
if __name__ == "__main__":
    args=get_parser()
    X_train, X_test, y_train, y_test, num_label =get_iris_data(args)
    y_train_bin=M_binary_encode_labels(y_train)
    
    num_feture=X_train.shape[1]
   
    qc,featN,weightN=build_circuit(X_train,args.ansatzReps,args.verb)
    num_weight=len(weightN)
    if args.verb>0 : print('M: full compact, num_weight=%d'%num_weight); print(qc)    
    
    backend = qk.Aer.get_backend('aer_simulator')
    qcT = qk.transpile(qc, backend=backend)
    print('M: transpiled exp=%s myRank=%d rndSeed=%d'%(args.expName,args.myRank,args.rndSeed));# print(qcT)

    weightsIni=init_weights(weightN,ampl=args.cobyla_rhobeg) 
    myHist=TrainHistory(args.verb)

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
    
    acc=M_evaluate(weightsOpt,txt='Pass1-end myRank=%d'%args.myRank)
    summary_line(acc)
    if args.myRank==0:    M_evaluate(myHist.bestW,txt='Pass1-best')    
    
    if not args.doublePass :  exit(0)

    # =============================================
    rhobeg= args.cobyla_rhobeg*2
    print('2nd COBYLA, rhobeg=%.2f ...'%rhobeg)

    # Reset loss history
    myHist=trainMemory(args.verb)
    result = minimize(fun=M_loss_function, 
                      x0=weightsOpt, 
                      args=(X_train, y_train_bin), 
                      method='COBYLA',                          
                      options={'maxiter': args.maxIter//2,'rhobeg': rhobeg})

    M_evaluate(weightsOpt,txt='Pass2-end')
    M_evaluate(myHist.bestW,txt='Pass2-best')
 
   
