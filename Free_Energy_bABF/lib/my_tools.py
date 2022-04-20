import numpy as np
import os,sys,re

def pbc_diff(v,Cell):
    return v-np.dot(Cell,np.floor(np.dot(np.linalg.inv(Cell),v.T)+0.5)).T # make sure no cross-boundary jumps.

class eam_info:
  def __init__(self,_pot_file,pair_style="eam/fs"):
    lc=0
    self.ele=[]
    if pair_style == "eam/fs":
        for line in open(_pot_file):
          lc+=1
          #print lc,line,re.sub("\t"," ",line).strip().split(" ")[1:],"\n\n\n"
          if lc == 4:
            for _ele in re.sub("\t"," ",line).strip().split(" ")[1:]:
              #print _ele
              if _ele != "":
                self.ele.append(_ele)
          if lc == 5:
            self.cutoff=np.fromstring(line.strip(),sep=' ')[-1]
          if lc == 6:
            self.mass=np.fromstring(line.strip()[:-3],sep=' ')[-2]
            self.lattice=np.fromstring(line.strip()[:-3],sep=' ')[-1]
            self.crystal=line.strip()[-3:]
          if lc > 6:
            break
    elif pair_style == "meam/spline":
        print("meam")
        # a bad hack: '#', 'Mo', '42', '95.94', '3.1674', 'bcc', '6.0'
        for _line in open(_pot_file):
            line = _line.strip().split(' ')
            self.ele.append(line[1])
            self.cutoff=float(line[6])
            self.mass=float(line[3])
            self.lattice=float(line[4])
            self.crystal=line[5]
            break

def loadin(_lmp, _file, _mass, _pot_file, _ele, pair_style='eam/fs'):
    _lmp.command('units metal')
    _lmp.command('atom_style atomic')
    _lmp.command('atom_modify map array sort 0 0')
    _lmp.command('read_data %s' % _file)
    _lmp.command('mass * %f' % _mass)
    _lmp.command('pair_style    %s' % pair_style)
    _lmp.command('pair_coeff * * %s %s' % (_pot_file, _ele))
    #_lmp.command('compute pe all pe')
    _lmp.command('variable pe equal pe')
    _lmp.command('compute pote all pe/atom')
    _lmp.command('compute cote all centro/atom bcc')
    _lmp.command('run 0')
    print("Loaded %s" % _file)

# Save array as LAMMPS.dat file
def save_lammps_file(_X,Cell,_file,mass=60.):
  _N = len(_X)
  head_f = open(_file, 'w')
  _header = "LAMMPS DATA FILE\n\n\n%d atoms\n\n1 atom types\n0. %10.10g xlo xhi\n0. %10.10g ylo yhi\n0. %10.10g zlo zhi\n%10.10g %10.10g %10.10g xy xz yz\n\nMasses\n\n1 %f\n\nAtoms\n" % (_N, Cell[0][0], Cell[1][1], Cell[2][2], Cell[0][1], Cell[0][2], Cell[1][2], mass)
  np.savetxt(_file, np.hstack((np.linspace(1, _N, _N).reshape(_N, 1), np.ones(_N).reshape(_N, 1), _X)), fmt="%d %d %10.10g %10.10g %10.10g", header=_header, comments='')

# Read array as LAMMPS.dat file
def read_lammps_file(_file,just_cell=False):
  Cell = np.zeros((3,3))
  c=0
  for line in open(_file):
    c+=1
    if c > 6 and c < 10:
      _line = np.fromstring(line,sep=' ')
      Cell[c-7][c-7] = _line[1] - _line[0]
    if c == 10:
      _line = np.fromstring(line,sep=' ')
      for j in range(3):
        Cell[j/2][(j+1)/2+1] = _line[j]
      break
  if just_cell == False:
    dat = np.loadtxt(_file,skiprows=17)
    return dat[np.argsort(dat[:,0])][:,[2,3,4]], Cell
  else:
    return Cell

def read_lammps_dump_file(_file):
  Cell = np.zeros((3,3))
  c=0
  for line in open(_file):
    c+=1
    if c > 5 and c < 9:
      _line = line.split(" ")
      Cell[c-6][c-6] = float(_line[1]) - float(_line[0])
      if len(_line) == 3:
        Cell[(c-6)/2][(c-5)/2+1] = float(_line[2])
    if c == 9:
      break
  dat = np.loadtxt(_file,skiprows=9)
  return dat[np.argsort(dat[:,0])][:,[1,2,3]], Cell
