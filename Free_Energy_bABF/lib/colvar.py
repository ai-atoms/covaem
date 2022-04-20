import numpy as np

def pbc_diff(v,Cell):
    return v-np.dot(Cell,np.floor(np.dot(np.linalg.inv(Cell),v.T)+0.5)).T # make sure no cross-boundary jumps.


class colvar:
    def __init__(self,Cell,X0,deltaX):
        self.Cell = Cell
        self.X0 = X0
        self.deltaX = deltaX
        self.norm2 = np.dot(deltaX.flatten(),deltaX.flatten())

    def evaluate(self,x):
        xdiff = pbc_diff(x - self.X0,self.Cell)
        cv = np.sum(xdiff*self.deltaX)/self.norm2
        dcv = self.deltaX/self.norm2
        return cv, dcv
