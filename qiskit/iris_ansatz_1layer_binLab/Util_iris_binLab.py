from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from time import time

#...!...!....................
def get_iris_data(args):
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    num_label=3  # for Iris data

    # Preprocess the data
    # Normalize data
    scaler = MinMaxScaler(feature_range=args.input_scale)
    X = scaler.fit_transform(X)

    if args.reverseFeatures:
        #X=X[::-1, :]
        X=X[:,::-1]
        if args.verb>0: print('get_iris_data: features order inversed')
    
    # Split into training and test sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_fraction,  random_state=args.rndSeed)
    
    if args.verb>0: print('Iris data split  train:',X_train.shape, ', test:',X_test.shape)
    return X_train, X_test, y_train, y_test ,num_label


#...!...!....................
def binary_cross_entropy_loss(Y_pred, Y_true):    
    #print('BCEL:',Y_pred.shape, Y_true.shape)
    # Adding a small value to prevent log(0)    
    loss= -np.mean(Y_true * np.log(Y_pred + 1e-9) + (1 - Y_true) * np.log(1 - Y_pred + 1e-9))
    return loss


#...!...!....................
def init_weights(weightN,ampl=0.1):
    nW=len(weightN)
    weights=np.random.rand(nW)*ampl 
    #weights=np.zeros(nW)
    return weights

    
#...!...!....................
def get_par_names(qc,txt='',verb=1):
    parNL= [param.name for param in qc.parameters]
    if verb>0: print(txt+' %d parNL:'%len(parNL),parNL)
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
class TrainHistory():  # book keeping class
    def __init__(self,verb=1):
        self.loss=[]      # Loss history
        self.bestIter=-1
        self.bestLoss=1e99
        self.tstart=time()
        self.verb=verb
        
    def add(self,val):
        #global weightCurrent
        self.loss.append(val)
        iIter=self.num_iter()
        if self.bestLoss> val :
            self.bestLoss=val
            self.bestIter=iIter
            self.bestW=self.weightCurrent
        self.elaTime=time()-self.tstart
        if self.verb>0 and iIter%10==0: print('iter=%d  loss=%.3f  elaT=%.2f min'%(iIter,val,self.elaTime/60.))
        
    def report(self,txt):
        lastIter=self.num_iter()
        print('\n%s ends, %d iterations, end-loss=%.3f'%(txt,lastIter,self.loss[-1]))
        if lastIter!=self.bestIter:
            print('   bestIter=%d, best loss=%.3f'%(self.bestIter,self.bestLoss))
        else:
            self.bestW=self.weightCurrent
            
    def num_iter(self): return  len(self.loss)
#...!...!....................
