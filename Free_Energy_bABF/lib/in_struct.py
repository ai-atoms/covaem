import numpy as np
import os
from scipy.interpolate import interp1d,splev

def pbc_diff(v,Cell):
    return v-np.dot(Cell,np.floor(np.dot(np.linalg.inv(Cell),v.T)+0.5)).T # make sure no cross-boundary jumps.
def file_parser(_file_name,forces=False):
    try:
        _X = np.loadtxt(_file_name)
        test_X = _X[:,[0,1,2]]
        if forces:
            test_F = _X[:,[3,4,5]]
            return test_X,test_F
        else:
            return test_X
    except ValueError:
        try:
            test_X=np.loadtxt(file_list[-1],skiprows=9)
            test_X=test_X[np.argsort(test_X[:,0])][:,[1,2,3]]
            return test_X
        except ValueError:
            print("Unknown file format!")
            return -1

class in_struct:
    def __init__(self,sc_file=None,mask_pc=100,thermal_expansion=1.,spline='cubic',forces=False):
        if sc_file is None:
            raise Error("SPECIFY SC FILE")

        _sc_folder = os.path.dirname(sc_file)
        """ Make Supercell matrix from SC_INFO file"""
        with open(sc_file) as _sc:
            head = [next(_sc).strip().split(" ") for x in range(5)]
            knot_pads = next(_sc).strip().split(" ")
            rel_file_list = [line.strip() for line in _sc]
        file_list = [os.path.join(_sc_folder,os.path.split(fl)[1]) for fl in rel_file_list]
        self.n_knots = len(file_list)

        self.Cell = np.zeros((3, 3))
        for i in range(3):
            self.Cell[i][i] = float(head[1 + i][1]) - float(head[1 + i][0])
            self.Cell[i // 2][(i + 1) // 2 + 1] = float(head[4][i])
        self.Cell *= thermal_expansion # Thermal Expansion

        iCell = np.linalg.inv(self.Cell)
        """ Supercell Loaded"""

        """ Load Knots """
        if forces:
            pX,pF = file_parser(file_list[0],forces=True)
            fX,fF = file_parser(file_list[-1],forces=True)
            pX *= thermal_expansion
            fF *= thermal_expansion
        else:
            pX = file_parser(file_list[0]) * thermal_expansion
            fX = file_parser(file_list[-1]) * thermal_expansion

        com = pX.mean(axis=0)
        deltaX = pbc_diff(fX-pX,self.Cell)
        dX_com = deltaX.mean(axis=0)
        deltaX -= dX_com

        self.struct = pX.copy().flatten()
        self.N = len(self.struct)//3
        self.delta = deltaX

        
class end_struct:
    def __init__(self,sc_file=None,mask_pc=100,thermal_expansion=1.,spline='cubic',forces=False):
        if sc_file is None:
            raise Error("SPECIFY SC FILE")

        _sc_folder = os.path.dirname(sc_file)
        """ Make Supercell matrix from SC_INFO file"""
        with open(sc_file) as _sc:
            head = [next(_sc).strip().split(" ") for x in range(5)]
            knot_pads = next(_sc).strip().split(" ")
            rel_file_list = [line.strip() for line in _sc]
        file_list = [os.path.join(_sc_folder,os.path.split(fl)[1]) for fl in rel_file_list]
        self.n_knots = len(file_list)

        self.Cell = np.zeros((3, 3))
        for i in range(3):
            self.Cell[i][i] = float(head[1 + i][1]) - float(head[1 + i][0])
            self.Cell[i // 2][(i + 1) // 2 + 1] = float(head[4][i])
        self.Cell *= thermal_expansion # Thermal Expansion

        iCell = np.linalg.inv(self.Cell)
        """ Supercell Loaded"""

        """ Load Knots """
        if forces:
            pX,pF = file_parser(file_list[0],forces=True)
            fX,fF = file_parser(file_list[-1],forces=True)
            pX *= thermal_expansion
            fF *= thermal_expansion
        else:
            pX = file_parser(file_list[0]) * thermal_expansion
            fX = file_parser(file_list[-1]) * thermal_expansion

        com = pX.mean(axis=0)
        deltaX = pbc_diff(fX-pX,self.Cell)
        dX_com = deltaX.mean(axis=0)
        deltaX -= dX_com

        tot_dist = np.linalg.norm(dX)
        self.struct = (fX-dX_com).copy().flatten()
        self.N = len(self.struct)//3
        self.delta = deltaX

        
