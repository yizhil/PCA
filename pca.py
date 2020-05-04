import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy.linalg as LN

def sigmoid(h):
    return 1/(1+np.power(np.e,-h))

def append1h(x):
    numData = x.shape[0]
    xfin = np.hstack([x,np.ones((numData,1))])
    return xfin

def append1v(x):
    numData = x.shape[1]
    xfin = np.vstack([x,np.ones((1,numData))])
    return xfin

def feedforward(x,layer1W,layer2W,layer3W):
    #add 1 as last column on x
    xFin = append1h(x)
    out1 = sigmoid(np.dot(layer1W,xFin.T))
    #add 1 as last row on out1
    out1Fin = append1v(out1)
    out2 = sigmoid(np.dot(layer2W,out1Fin))
    #add 1 as last row on out2
    out2Fin = append1v(out2)
    out3 = sigmoid(np.dot(layer3W,out2Fin))
    return {'lay1':out1,'lay2':out2,'lay3':out3}

def commonDirection(allData):
    covMat=np.matmul(allData.T,allData)
    w,v = LN.eig(covMat)
    maxInd=np.argwhere(w==max(w))
    component=v[:,maxInd[0]].real
    return component

def dataProjection(dataVector,component):
    return np.dot(dataVector,component)

def componentRemoval(xOld,component,z):
    return xOld - z*component

def findComponents(allData, n):
    allDataFin = allData.copy()
    numData,numFeat = allData.shape
    topComponents = np.zeros((n,numFeat))
    #loop n times to extract n components
    for i in range(n):
        component = commonDirection(allDataFin)
        topComponents[i] = component.T
        #loop over datasets
        for j in range(numData):
            z = dataProjection(allDataFin[j],component)
            allDataFin[j] = componentRemoval(allDataFin[j],component.T,z)
    return topComponents

