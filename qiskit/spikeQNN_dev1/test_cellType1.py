#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
construct 1 input quantum neuron with
INPUT:
- 1 input 'p_inp' \in [0,1]
- 1 weights 'w'  \in [-1/2,1/2]
- 1 bias 'b'  \in [-1/2,1/2] 
OUTPUT:
- 1 meas bit : cr_out
  probability 'p_out' \in [0,1]
      p_out= (1 + b + w*p_inp)/2   
MATH:
- Rx angle encoding:   cos(theta)= -b -w*p_inp
   
'''

import numpy as np
from pprint import pprint
import os
from time import time
from qiskit import   Aer, transpile
from Util_spikeQNN import circ_cellType1
from qiskit import QuantumCircuit,ClassicalRegister, QuantumRegister

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-i','--numSample', default=5, type=int, help='num of images packed in to the job')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=5000, help="shots per circuit")
    parser.add_argument("--expName",  default='spikeT1',help='(optional) replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument('-b','--backend',default='aer',   help="backend for transpiler" )
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
    parser.add_argument("--outPath",default='out/',help="all outputs from  experiment")
  
    args = parser.parse_args()
    # make arguments  more flexible
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)

    return args

#...!...!....................
def configCT1(args):
    cf={}  # config
    cf['qr']=QuantumRegister(1) # must be reset
    cf['cro']=ClassicalRegister(1,'out')  # output 
    cf['theta_name']='the'
    cf['cell_type']='CT1'
    return cf
    
#...!...!....................
def construct_input_CT1(n_img):    

    # generate user data , float random
    eps=1e-4 # just in case
    Pinp = np.random.uniform(    eps, 1.-eps, size=(n_img))  # input
    W = np.random.uniform(-0.5+eps, 0.5-eps, size=(n_img))  # weight
    B = np.random.uniform(-0.5+eps, 0.5-eps, size=(n_img))  # bias

    X= - Pinp*W - B
    Xang=np.arccos(X) # encoded  data
    Pout= (1-X)/2. # true output
    
    print('input Pinp=',Pinp.shape,'\n',repr(Pinp[...,:3].T))
    outD={'p_inp':Pinp,'weight':W,'bias':B,'the':Xang,'p_out':Pout}

    # Check if the array contains any NaN values
    has_nan = np.isnan(Xang).any()
    if has_nan:
        print('NaN detected in CICT1, dump and abort')
        pprint(outD)
        print('NaN detected in CICT1, ABORT')
        exit(99)
    return outD

#...!...!....................
def eval_type1(ic,probsBL,inpD):
    #print('\n eval_3input')
    mprob=probsBL[ic]['1']

    tprob=inpD['p_out'][ic]
    #print('Y:',Y.T)
    
    res=tprob-mprob
    print('Eval: circT1 ic=%d   tprob=%7.3f  mprob=%7.3f   res=%7.3f '%(ic,tprob,mprob,res))


    return


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    cfgCT1=configCT1(args)
    pprint(cfgCT1)

    expD=construct_input_CT1(n_img =args.numSample)
    print('M:rnd inp:',sorted(expD))
    if args.verb>1: pprint(expD)
    
    #....  circuit generation .....
    qcP,paramD=circ_cellType1(**cfgCT1)
    
    print(qcP.draw(output='text',idle_wires=False))  # skip ancilla


    print('M: acquire backend:',args.backend)
    assert args.backend=='aer'
    backend = Aer.get_backend('aer_simulator')
    qcT = transpile(qcP, backend=backend, optimization_level=3, seed_transpiler=44)
    print(qcT.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla

    
    #... collect needed params and clone circuits
    qcEL=[ None for i in range(args.numSample) ]
    for i in range(args.numSample):
        cparamD={ paramD[xx]:expD[xx][i] for xx in paramD}
        if i<2: print('M:mapped params',i, cparamD,'\n')
        # Bind the values to the circuit
        qc1=qcT.assign_parameters(cparamD)
        qcEL[i]=qc1

    print(qc1.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla
    nCirc=len(qcEL)

    print('job started, nCirc=%d  nq=%d  shots/circ=%d at %s ...'%(nCirc,qcEL[0].num_qubits,args.numShot,backend))
    T0=time()
    job = backend.run(qcEL,shots=args.numShot)
    result=job.result()
    elaT=time()-T0
    print('M:  ended elaT=%.1f sec'%(elaT))
    
    probsBL=[{'0':0,'1':0}  for ic in range(nCirc)] 
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        #print('dump ic=',ic); pprint(counts)
        for x in counts:
            probsBL[ic][x]+=counts[x]/args.numShot
        if ic<10:  eval_type1(ic,probsBL,expD)
    pprint(probsBL)

    print('M:done')
