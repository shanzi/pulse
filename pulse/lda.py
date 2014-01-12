from pylab import *
from numpy import *
from scipy import linalg as la

def lda(data,labels,redDim):

    # Centre data
    data -= data.mean(axis=0)
    nData = shape(data)[0]
    nDim = shape(data)[1]
    
    Sw = zeros((nDim,nDim))
    Sb = zeros((nDim,nDim))
    
    C = cov(transpose(data))
    
    # Loop over classes
    classes = unique(labels)
    for i in range(len(classes)):
        # Find relevant datapoints
        indices = squeeze(where(labels==classes[i]))
        d = squeeze(data[indices,:])
        classcov = cov(transpose(d))
        Sw += float(shape(indices)[0])/nData * classcov
        
    Sb = C - Sw
    # Now solve for W
    # Compute eigenvalues, eigenvectors and sort into order
    #evals,evecs = linalg.eig(dot(linalg.pinv(Sw),sqrt(Sb)))
    evals,evecs = la.eig(Sw,Sb)
    indices = argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    w = evecs[:,:redDim]
    #print evals, w
    
    newData = dot(data,w)
    return newData,w

