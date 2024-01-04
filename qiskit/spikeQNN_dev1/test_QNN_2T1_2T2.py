#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
construct QNN with 2 inputs, 2 layers, 2 outputs
(layer1)
- 2 CT1 1q cells labeled 'A,B' with 
    input: 'p_inp' \in [0,1], 'w,b'  \in [-1/2,1/2]
    output  'cro'  classical binary register
    math: p_out= (1+b+w*p_inp)/2

- 2 CT2 3q cells labeled 'C,D' with 
   input: 2 bits cr_inp[2],  w[2],b  \in [0,1]
   output:  3 per qubit bits cr_out[3]
      and post-processed 1 bit  1-'000'
   math:  p_out= 1 - (1-w0*p_inp0) (1-w1*p_inp1) (1-b)   
'''

import numpy as np
from pprint import pprint
import os
import copy
from time import time
from qiskit import   Aer, transpile
from qiskit import QuantumCircuit,ClassicalRegister, QuantumRegister
from Util_spikeQNN import circ_cellType1,  circ_cellType2
from test_cellType1 import construct_input_CT1
from qiskit.result.utils import marginal_distribution

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-i','--numSample', default=5, type=int, help='num of images packed in to the job')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=10000, help="shots per circuit")
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
def buildQNN_2xT1_1xT2():
    
    nq=6
    qr=QuantumRegister(nq) # must be reset
    cri=ClassicalRegister(2,'inp')  # input
    croc=ClassicalRegister(3,'outc')  # output C
    crod=ClassicalRegister(3,'outd')  # output D
    qcP = QuantumCircuit(qr,cri,crod,croc,name='main')
    cfg={}  # full config
    expPar={}
    
    cf={}#----------- config node A
    cf['qr']=[qr[0]]
    cf['cro']=[cri[0]]
    cf['theta_name']='theA'
    cfg['A']=cf
    qcA,parA=circ_cellType1(**cf)
    expPar.update(parA)
    #1print(qcA.draw(output='text',idle_wires=False,cregbundle=False))


    cf={}#----------- config node B
    cf['qr']=[qr[1]]
    cf['cro']=[cri[1]]
    cf['theta_name']='theB'
    cfg['B']=cf
    qcB,parB=circ_cellType1(**cf)
    expPar.update(parB)
    #1print(qcB.draw(output='text',idle_wires=False,cregbundle=False))

    cf={}#----------- config node C
    cf['qr']=qr[:3]
    cf['cri']=cri
    cf['cro']=croc
    cf['theta_name']='theC'
    cfg['C']=cf
    qcC,parC=circ_cellType2(**cf)
    expPar.update(parC)
    #1print(qcC.draw(output='text',idle_wires=False,cregbundle=False))

    cf={}#----------- config node D
    cf['qr']=qr[3:]
    cf['cri']=cri
    cf['cro']=crod
    cf['theta_name']='theD'
    cfg['D']=cf
    qcD,parD=circ_cellType2(**cf)
    expPar.update(parD)
    
    #.... assemble all circuits together
    qcP.compose(qcA, inplace=True)
    qcP.compose(qcB , qubits=cfg['B']['qr'], clbits=cfg['B']['cro'] , inplace=True)

    crL=[ x for x in cri ]
    for x in croc : crL.append(x) 
    qcP.compose(qcC, qubits=cfg['C']['qr'], clbits=crL, inplace=True)
    
    crL=[ x for x in cri ]
    for x in crod : crL.append(x) 
    qcP.compose(qcD, qubits=cfg['D']['qr'], clbits=crL ,inplace=True)
 
    pprint(expPar)
    return qcP,expPar
    
#...!...!....................
def configCT2(args):
    cf={}  # config
    nq=3
    cf['qr']=QuantumRegister(nq) # must be reset
    cf['cri']=ClassicalRegister(2,'inp')  # input
    cf['cro']=ClassicalRegister(nq,'out')  # output 
    cf['theta_name']='the'
    cf['cell_type']='CT2'
    return cf

#...!...!....................
def construct_input_CT2(n_img, Pinp):    
    n_inp=2
    # generate user data , float random
    eps=1e-4 # just in case
    W = np.random.uniform(eps, 1.-eps, size=(n_img,n_inp))  # weight
    B = np.random.uniform(eps, 1.-eps, size=(n_img))  # bias

    if 0:
        assert n_img==1
        W=np.array([[0.7, 0.28843403]])
        B=np.array([0.68193204])
    X=np.zeros((n_img,n_inp+1))  # for all 3 qubits
    X[:,:n_inp]=1-2*W
    X[:,2]=1-2*B    
    Xang=np.arccos(X) # encoded cell weights & bias
    
    # predict output based on  probs of input bits
    Pout=np.zeros_like(X)
    Pout[:,0]= Pinp[0]*W[:,0]
    Pout[:,1]= Pinp[1]*W[:,1]
    Pout[:,2]= B

    #... compute success for multiple Bernoullie trials
    Pm0=1-Pout
    pr0=np.prod(Pm0,axis=1)
    #print('ppp', Pm0.shape, pr0.shape)
    Pout_mber=1-pr0
    
    outD={'weight':W,'bias':B,'the':Xang,'X':X,'p_out_bits':Pout, 'p_out_mber':Pout_mber}
    
    # Check if the array contains any NaN values
    has_nan = np.isnan(Xang).any()
    if has_nan:
        print('NaN detected in CICT2, dump and abort')
        pprint(outD)
        print('NaN detected in CICT2, ABORT')
        exit(99)
    return outD

#...!...!....................
def construct_input_2xT1_2xT2():    
    n_img=args.numSample

    expOut={}    
    expA=construct_input_CT1(n_img)
    expB=construct_input_CT1(n_img)
    Pouta=expA['p_out']
    Poutb=expB['p_out']
    
    expC=construct_input_CT2(n_img,[Pouta,Poutb])
    expD=construct_input_CT2(n_img,[Pouta,Poutb])

    expOut['theB']=expB['the']
    expOut['theA']=expA['the']
    expOut['theC']=expC['the']
    expOut['theD']=expD['the']

    expOut['p_out_mberC']=expC['p_out_mber']
    expOut['p_out_mberD']=expD['p_out_mber']
   
    print('Dump param all:',expOut)
    return expOut

#...!...!....................
def eval_2xT1_2xT2(ic,probsBL,inpD,nCell):
    print('\n eval_2xT1_2xT2 ic=',ic)
    probsB=probsBL[ic]
    #print('ic=',ic); pprint(probsB)

    tprob=np.array([inpD['p_out_mberC'][ic],inpD['p_out_mberD'][ic]])
    mprob=np.array([probsB[j]['1']  for j in range(nCell) ])
    #print('tprob:',tprob); print('mprob:',mprob)
    res=tprob-mprob
    for j in range(nCell):
        print('Eval:  M-bernoullie circ ic=%d j=%d  tprob=%7.3f  mprob=%7.3f   res=%7.3f '%(ic,j,tprob[j],mprob[j],res[j]))

    

    return


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    if 0:
        cfgCT2=configCT2(args)
        pprint(cfgCT2)
        qc,paramD=circ_cellType2(**cfgCT2)
        print(qc.draw(output='text',idle_wires=False,cregbundle=False)) 
        mmm89
    
   
    #....  circuit generation .....
    qcP,paramD=buildQNN_2xT1_1xT2()
    print('M: paramD:',sorted(paramD))
    print(qcP.draw(output='text',idle_wires=True,cregbundle=False))
    
    expD=construct_input_2xT1_2xT2()
    print('M:rnd inp:',sorted(expD))
    if args.verb>1: pprint(expD)

    print('M: acquire backend:',args.backend)
    assert args.backend=='aer'
    backend = Aer.get_backend('aer_simulator')
    qcT = transpile(qcP, backend=backend, optimization_level=3, seed_transpiler=44)
    #1print(qcT.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla

    
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

    nCell=2  # number of cells in the last layer

    # Creating a list of independent dictionaries for cell counts
    nullCnt = [{'0': 0., '1': 0.} for _ in range(nCell)]

    print('nullCnt:',nullCnt)
    probsBL = [copy.deepcopy(nullCnt) for _ in range(nCirc)]
    
    for ic in range(nCirc):
        counts = result.get_counts(ic)
        #print('dump ic=',ic); pprint(counts) ; rrr
        for x in counts:
            xL=x.split(' ')
            for j in range(nCell):
                mbit=xL[j]
                y='0' if mbit=='000' else '1'
                #print('xxx',x,j,mbit,y)            
                probsBL[ic][j][y]+=counts[x]/args.numShot
        #print('myCnt:',probsBL[ic])
        
        if ic<10:  eval_2xT1_2xT2(ic,probsBL,expD,nCell)
    #pprint(probsBL)
    
    print('M:done')
