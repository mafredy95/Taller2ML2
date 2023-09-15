import numpy as np

class SVD:

    def __init__(self):
        pass

    def fit_transform(self,M):
    
        
        #M=UxDxVT

        #Calculating eigen-vectors corresponding to V matrix
        MTxM=M.T@M
        eVal1,eVec1=np.linalg.eigh(MTxM)
        cols_new_order=np.argsort(eVal1)[::-1]
        self.VT=eVec1[:,cols_new_order].T

        #Calculating eigen-vectors corresponding to U matrix
        MxMT=M@M.T
        eVal2,eVec2=np.linalg.eigh(MxMT)
        cols_new_order=np.argsort(eVal2)[::-1]
        self.U=eVec2[:,cols_new_order]
    

        #Calculating eigen-values corresponding to S matrix
        n,m=M.shape
        if n>m:
            S=np.diag(np.sqrt(np.sort(np.abs(eVal1)))[::-1])
            zeros=np.zeros((n-m,m))
            self.S=np.concatenate((S, zeros), axis=0)

        elif n<m:
            S=np.diag(np.sqrt(np.sort(np.abs(eVal2)))[::-1])
            zeros=np.zeros((m-n,n))
            self.S=np.concatenate((S, zeros.T), axis=1)

        else:
            self.S=np.diag(np.sqrt(np.sort(np.abs(eVal2)))[::-1])
            

        return [self.U,self.S,self.VT]
