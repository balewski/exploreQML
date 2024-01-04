from qiskit.circuit import Parameter,ParameterVector
from qiskit import QuantumCircuit

#...!...!....................
def circ_cellType1(qr,cro,theta_name,**kwargs):
    print('CT1:  qr,cro,theta_name:',qr,cro,theta_name)
    qc = QuantumCircuit(qr,cro)
    thetaP=Parameter(theta_name)
    qc.rx(thetaP,qr)
    qc.measure(qr,cro)
    qc.reset(qr)
    parD={theta_name:thetaP}
    return qc,parD
   
#...!...!....................
def circ_cellType2(qr,cri,cro,theta_name,**kwargs):
    print('CT2:  qr,cri,cro,theta_name:',qr,cri,cro,theta_name)
    nq=len(qr) ; assert nq==3
    qc = QuantumCircuit(qr,cri,cro)
    thetaP=ParameterVector(theta_name,length=nq)
   
    for i in range(2):
        qc.ry(thetaP[i]/2,qr[i])
        
        with qc.if_test((cri[i],1)):  # good for scaling
            qc.x (qr[i])
        qc.ry(-thetaP[i]/2,qr[i])
        with qc.if_test((cri[i],1)):  # good for scaling
            qc.x (qr[i])

        #qc.rz(-thetaP[i]/2,qr[i])
        #qc.x (qr[i])
    qc.rx(thetaP[2],qr[2])
    qc.barrier()
    for i in range(nq):
        qc.measure(qr[i],cro[i])
    qc.reset(qr) 
    parD={theta_name:thetaP}
    return qc,parD
   
