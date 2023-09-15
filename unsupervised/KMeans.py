import numpy as np

class my_KMeans:
    def __init__(self,k=2,n=10):
        self.k=k
        self.n=n

    def fit_transform(self,X):
        X=X.copy()
        Xshape=X.shape
        rows,cols=Xshape
        
        C=np.zeros(rows)

        aux=True
        for n in range(self.n):
            if aux:
                while len(np.unique(C))!=self.k:
                    Z=self.init_Z(cols,X)
                    for row,x in enumerate(X):
                        C[row]=self.assignment(x,Z)
                aux=False
            else:
                for row,x in enumerate(X):
                        C[row]=self.assignment(x,Z)

            for k in range(self.k):
                indices = np.where(C == k)
                Gk=X[indices]
                Z[k,:]=np.mean(Gk,axis=0)

        self.centroides=Z
        return C

    def init_Z(self,cols,X):
        Z=np.zeros((self.k,cols))
        for k in range(self.k):
            for i in range(cols):
                min=np.min(X[:,i:i+1])
                max=np.max(X[:,i:i+1])
                Z[k,i]=np.random.uniform(min,max)
        return Z

    def assignment(self,X,Z):
        distances=np.zeros(self.k)   
        for k in range(self.k):
            distances[k]=np.linalg.norm(X-Z[k:k+1,:])
        best_cluster=np.argsort(distances)[0]
        return best_cluster
    
       