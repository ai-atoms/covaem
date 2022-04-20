#!/usr/bin/env python
"""
Jacopo BAIMA 2022
Free energy integration using Bayesian ABF and AutoEncoder extraction of collective variables
WARNING: predisposed for MPI but current version should be used in SERIAL
"""
import os,sys,socket,time,shutil
import math
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d,BSpline, CubicSpline
from scipy.integrate import cumtrapz
from mpi4py import MPI
from optparse import OptionParser
tf.config.set_visible_devices([], 'GPU') # comment to use GPU

""" MPI """
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# New communicator to run size//1 lammps() objects concurrently
split = comm.Split(color=rank)
""" MPI """

rt_dir = os.path.abspath(os.path.split(sys.argv[0])[0])
#pot_loc = os.environ['HOME']+"/POTS/" # where the EAM potentials are
pot_loc = rt_dir # where the EAM potentials are
sys.path.insert(0,os.path.join(rt_dir,'lib/'))
from my_tools import *
from lammps import lammps
from timestep_bayes import integrator
from in_struct import in_struct
from colvar import colvar
from train_ae import cv_autoenc

""" *************** command line parser /  system parameters *************** """
parser = OptionParser()
parser.add_option("-t", "--temperature", dest="temperature", default=200., help="Temperature in Kelvin")
parser.add_option("-r", "--runid", dest="runid", default=0, help="Run ID")
parser.add_option("-s", "--steps", dest="steps", default=2800, help="MD timesteps per bABF run")
parser.add_option("-b", "--bins", dest="bins", default=50, help="Number of bins for free energy discretization")
parser.add_option("-i", "--input", dest="input", default=None, help="Input File", type=str)


(options, args) = parser.parse_args()

default_sc_file = os.path.join(rt_dir,"struc/SC_INFO")
default_pot_file_name = "pot.fs"

sc_file = default_sc_file
pot_file_name = default_pot_file_name
TEMP = float(options.temperature)
RUN = int(options.runid)
STEPS = int(options.steps)
BINS = int(options.bins)
DUMP_OUTPUT = True
STRESS = None
MASK_PC = 100 # Include MASK_PC% of atoms from path
scale=1.
expansion = None
fields = ['temperature', 'bins' ,'steps', 'potential', 'system', 'expansion', 'dump']
if not options.input is None:
    for _line in open(str(options.input)):
        line=_line.strip().split(" ")
        if line[0] == fields[0]:
            if TEMP == float(parser.option_list[1].default):
                TEMP = float(line[1])
        elif line[0] == fields[1]:
            if BINS == float(parser.option_list[4].default):
                BINS = int(line[1])
        elif line[0] == fields[2]:
            if STEPS == float(parser.option_list[3].default):
                STEPS = int(line[1])
        elif line[0] == fields[3]:
            pot_file_name = str(line[1])
        elif line[0] == fields[4]:
            sc_file = os.path.join(rt_dir,str(line[1]))
        elif line[0] == fields[5]:
            expansion=[]
            for i in range(1,len(line)):
                expansion.append(float(line[i]))
        elif line[0] == fields[6]:
            DUMP_OUTPUT = bool(float(line[1]))

if rank == 0:
    print("sc_file",sc_file)
    print("\n\nINPUT OPTIONS:\n")
    print(fields[0],":",TEMP,"\n")
    print(fields[1],":",BINS,"\n")
    print(fields[2],":",STEPS,"\n")
    print(fields[3],":",pot_file_name,"\n")
    print(fields[4],":",sc_file,"\n")
    print(fields[5],":",expansion,"\n")
 
pot_file = os.path.join(pot_loc, pot_file_name)
pair_style = 'eam/fs'
if pot_file.split('.')[-1] == 'spline':
    pair_style = 'meam/spline'

pot = eam_info(pot_file,pair_style=pair_style) # ele[], mass, lattice, cutoff as members
lattice_constant = pot.lattice
if not expansion is None:
    for c in range(len(expansion)):
        lattice_constant += expansion[c]*(TEMP**(c+1))
scale = lattice_constant / pot.lattice
if rank == 0:
    print("lattice constant",":",lattice_constant,"(%1.2g%% expansion)" % float(100.*(scale-1)),"\n")

""" *************** END command line parser *************** """

""" *************** system parameters *************** """
TOT_TIME = STEPS	# Total simulation steps
if rank == 0:
    _folder_name = sc_file.split("/")[-2]+"_%d" % TEMP
    tmp_count = 0
    dump_folder = _folder_name+"_%d" % tmp_count
    while os.path.isdir(dump_folder):
        tmp_count += 1
        dump_folder = _folder_name+"_%d" % tmp_count    
    os.mkdir(dump_folder)
    struct_folder = os.path.join(dump_folder,"structures/")
    forces_folder = os.path.join(dump_folder,"forces/")
    os.mkdir(struct_folder)
    os.mkdir(forces_folder)
else:
    dump_folder = None
    struct_folder = None
    forces_folder = None
    
comm.Barrier()

dump_folder = comm.bcast(dump_folder,root=0)
struct_folder = comm.bcast(struct_folder,root=0)
forces_folder = comm.bcast(forces_folder,root=0)
runtime_file = os.path.join(dump_folder,"runtime_dump_%d_%d_%d_%s" % (RUN,size,TOT_TIME,TEMP))
start_file = os.path.join(dump_folder,"_start_temp.dat")
results_file = os.path.join(dump_folder,"MPI_%d_RESULTS_%d_%d_%d" % (RUN,int(TEMP), TOT_TIME,size))
histo_file = os.path.join(dump_folder,"MPI_%d_HISTO_%d_%d_%d" % (RUN,int(TEMP), TOT_TIME,size))
ofl_file = os.path.join(dump_folder,"mpi_%d_dev_%d_%d" % (RUN,size,int(TEMP)))
fin_file = os.path.join(dump_folder,"mpi_%d_final_%d_%d" % (RUN,size,int(TEMP)))

neb_file =  os.path.join(dump_folder,"ZEROK_%d" % (RUN))

SCREEN_DUMP_FREQ = 100 # output every SCREEN_DUMP_FREQ timesteps
if STEPS <= 100:
    SCREEN_DUMP_FREQ = 5 # extra frequent if short test run

THERM_STEPS = 1000 # Thermalization steps - typically fine

print("THERM_STEPS",THERM_STEPS)


kB = 1.38 / 1.6 * 0.0001  # kB in eV/K


""" *************** End System Parameters *************** """


""" start up MD """
start = in_struct(sc_file=sc_file,mask_pc=MASK_PC,thermal_expansion=scale) #read start structure

Cell = start.Cell

_N = start.N
_N_M = _N
deltaX = start.delta

comm.Barrier() # wait

lmp = lammps(cmdargs=["-screen","none","-log","none"])


if rank == 0:
  save_lammps_file(start.struct.reshape(_N,3),Cell,start_file,mass=pot.mass) # Save as lammps.dat file
comm.Barrier()


loadin(lmp, start_file, pot.mass, pot_file, pot.ele[0], pair_style=pair_style) # list of standard commands



# ctypes pointers to LAMMPS objects
_x = np.ctypeslib.as_array(lmp.extract_atom("x",3).contents,shape=(_N,3))
_v = np.ctypeslib.as_array(lmp.extract_atom("v",3).contents,shape=(_N,3))
_f = np.ctypeslib.as_array(lmp.extract_atom("f",3).contents,shape=(_N,3))



_x[:] = start.struct.reshape((_N,3))
X0 = _x.copy()
V0 = _v.copy()
F0 = _f.copy()
lmp.command('run 0') # To make neighbour lists etc
""" END start up MD """

""" Monitoring / analysis """
if DUMP_OUTPUT:
    rw = open(runtime_file+"_%d" % rank,'w') # logfile

""" END Monitoring / analysis """


""" MD """
_step = np.zeros((_N,3)) # dX vector
lmp.command("run 0")
hc=np.zeros(BINS)

block_rad = 0.6 * np.sqrt(.75) * lattice_constant

en_file = os.path.join(dump_folder,"freen.dat")
enf = open(en_file,'w')

prob_file = os.path.join(dump_folder,"prob.dat")
probf = open(prob_file,'w')


### intitialize parameters
dt = 0.001

sigma = 1./float(BINS)

SMOOTH_BINS = float(BINS)
spring = (TEMP*kB)*(SMOOTH_BINS*SMOOTH_BINS)
spring0 = spring
gamma = TEMP*dt/pot.mass

spring_scheduler = False
prob_scheduler = True
schedule_steps = TOT_TIME #10000 #int(TOT_TIME/10)
tau = 3*TEMP*TEMP/(float(BINS)*schedule_steps)
binsmin = False
nbinsmin = 0.+tau

### initialize saved and bin-discretized quantities
n_bins = np.zeros(BINS,dtype='int32')
avg_force = np.zeros(BINS)
free_en_endbin = np.zeros(BINS+1)
free_en_midbin = np.zeros(BINS)
avg_force_num = np.zeros(BINS)
avg_force_den = np.full(BINS,tau)
cond_prob = np.full(BINS,1./float(BINS))
log_prob_num = np.zeros(BINS)
pos_bin = (np.arange(BINS)+0.5)/float(BINS)
biased_en = np.zeros(BINS)
pos_struc = np.zeros(TOT_TIME,dtype='int32')

### generate collective variable guess epsi0(r)
cv = colvar(Cell,X0,deltaX)

epsi, depsi = cv.evaluate(X0)
print("start",epsi,depsi[0])
epsi, depsi = cv.evaluate(X0+deltaX)
print("end",epsi,depsi[0])

### define integrator 
integrate = integrator(lmp,Cell,_x,_v,_f,mass=pot.mass,\
                            gamma=gamma,temperature=TEMP,\
                            seed=(rank+1)*int(np.random.uniform(1000)),\
                            block_rad=block_rad, cosmin_mask=None,dt=dt,\
                            spring=spring)
_time = time.time()


biasf = 0.
for t in range(THERM_STEPS): 
    epsi, depsi = cv.evaluate(_x)
    _x,_f = integrate.eABF_brownian(depsi,biasf)
    log_prob_num = -((spring/2)*(epsi-pos_bin)**2)/(TEMP*kB)
    log_prob_num = log_prob_num-np.max(log_prob_num) ##avoid overflow
    prob_den = np.sum(np.exp(log_prob_num))
    cond_prob = np.exp(log_prob_num)/prob_den
    springf = -spring*(epsi-pos_bin)
    biasf = np.sum(springf*cond_prob)
    
stats_file = os.path.join(dump_folder,"stats.dat")
statsf = open(stats_file,'w')
dev_file = os.path.join(dump_folder,"devf.dat")
devf = open(stats_file,'w')


maxepsichange = 0.
maxxchange = 0.
springf = 0.
biasf = 0.

if spring_scheduler:
  spring = 0.

for t in range(TOT_TIME): 
    oldepsi = epsi.copy()
    oldx = _x.copy()
    
    epsi, depsi = cv.evaluate(_x)
    if epsi < 1. and epsi > 0.:
      thisbin = math.floor(epsi*BINS)
      n_bins[thisbin] += 1

    _x,_f = integrate.eABF_brownian(depsi,biasf)

    log_prob_num = -((spring/2)*(epsi-pos_bin)*(epsi-pos_bin)-free_en_midbin)/(TEMP*kB)
    log_prob_num = log_prob_num-np.max(log_prob_num) ##avoid overflow
    prob_den = np.sum(np.exp(log_prob_num))
    cond_prob = np.exp(log_prob_num)/prob_den
    if prob_scheduler:
      cond_prob_w = np.minimum(cond_prob*t/schedule_steps,cond_prob)
    else:
      cond_prob_w = cond_prob
    
    springf = -spring*(epsi-pos_bin)
    avg_force_num += springf*cond_prob_w
    avg_force_den += cond_prob_w
    avg_force = avg_force_num/avg_force_den
    if binsmin: ### don't count avg_force if it has never passed from there
      avg_force = avg_force*(avg_force_den>nbinsmin) 
    for i in range(1,BINS+1):  
      free_en_endbin[i] = free_en_endbin[i-1]+(avg_force[i-1]/float(BINS))
      free_en_midbin[i-1] = (free_en_endbin[i]+free_en_endbin[i-1])/2.

    if spring_scheduler:
      spring = min(spring0*t/schedule_steps,spring0)
    springf = -spring*(epsi-pos_bin)
    biasf = np.sum(springf*cond_prob)#/float(BINS)
      
    epsichange = epsi-oldepsi
    xchange = np.mean(np.abs(pbc_diff(_x-oldx,Cell)))
    xch_rms = np.sqrt(np.mean(np.square(pbc_diff(_x-oldx,Cell))))

    maxepsichange = max(abs(epsichange),maxepsichange)
    maxxchange = max(abs(xchange),maxxchange)
    
    if t%100000 == 0:
      print('step ',t)
    if t%100 == 0:
      statsf.write(str(t)+' '+str(epsi)+' '+str(epsichange)+' '+str(xchange)+' '+str(xch_rms)+' '+str(biasf)+' '+'\n')#+' '+str(biasf)+' '+
    if t%10000 == 0:
      probf.write('Initial run step '+str(t)+' pos '+str(epsi)+' \n')
      for i in range(BINS):
        probf.write(str(i)+' '+str(cond_prob[i])+' '+str(avg_force_den[i])+' \n')
    #save all structures, save position vector
    if t%10 == 0:
      postest = pbc_diff(_x-X0,Cell)[0,:]
      deltatest = deltaX[0,:]
      pos1 = np.dot(deltatest,postest)/np.dot(deltatest,deltatest)
      deltaorth = postest-pos1*deltatest
      posorth = np.sqrt(np.dot(deltaorth,deltaorth)/np.dot(deltatest,deltatest))
      devf.write(str(t)+' '+str(pos1)+' '+str(posorth)+' '+'\n') 
      t10 = t//10
      struct_file = os.path.join(struct_folder,"struct_%d" % t10)
      xdiff = _x - X0
      xsave = _x.copy()
      ### this works only for 90° angles cell
      for j in range(_N):
        for k in range(3):
          if xdiff[j,k]>Cell[k,k]/2:
            xsave[j,k]-=Cell[k,k]
          if xdiff[j,k]<-Cell[k,k]/2:
            xsave[j,k]+=Cell[k,k]
      xsave.tofile(struct_file)
print('maxepsichange',maxepsichange,'maxxchange',maxxchange)

statsf.close()
print("end first run")


for i in range(BINS):
  print(i,avg_force[i],n_bins[i],avg_force_den[i])

free_energy_bins = np.zeros(BINS+1)
fen = 0. 
for i in range(BINS+1):  
  print(i,free_en_endbin[i])
  enf.write(str(i)+' '+str(free_en_endbin[i])+' \n')
enf.write('&')

cv_new = [cv]
comm.Barrier() # wait

###############################################################
####### start looping on CV, code is mostly as above ##########
for icv in range(1,4):
  ####### train autoencoder ############

  ndata = (TOT_TIME)//10
  cv_new.append(cv_autoenc(Cell,X0,deltaX,ndata,_N,dump_folder,struct_folder))
  cv_true = colvar(Cell,X0,deltaX)

  ####### run convergence test
  ntest = 25
  xtest = cv_new[icv].generate(ntest)
  testval = 0.
  print("test")
  for itest in range(ntest):
    epsi, depsi = cv_new[icv].evaluate(xtest[itest])
    epsi_old, depsi_old = cv_new[icv-1].evaluate(xtest[itest])
    dep = depsi.flatten()
    depo = depsi_old.flatten()
    testval += 1-(np.dot(dep,depo)/np.sqrt(np.dot(dep,dep)*np.dot(depo,depo)))
    print(itest,testval)
    #print(dep)
    #print(depo)
  testval = testval/ntest
  print("convergence test",testval)
  if testval < 0.01:
    print("converged")
    
  testval = 0. 
  if icv > 1:
    for itest in range(ntest):
      xnew1 = cv_new[icv].autoencode(xtest[itest])
      xnew2 = cv_new[icv-1].autoencode(xtest[itest])
      testval += np.sqrt(np.mean(np.square(xnew1-xnew2)))
    testval = testval/ntest
    print("convergence test 2",testval)

  ######################################            

  _x[:] = X0
  _v[:] = V0
  _f[:] = F0
  lmp.command("run 0")

  ### intitialize parameters
  dt = 0.001

  spring_scheduler = False
  prob_scheduler = True
  schedule_steps = TOT_TIME 
  tau = 3*TEMP*TEMP/(float(BINS)*schedule_steps)
  binsmin = False
  nbinsmin = 0.+tau

  ### use previous simulation as prior for average force 
  avg_force_num[:] = avg_force*tau
  avg_force_den[:] = tau
  
  ### initialize saved and bin-discretized quantities
  n_bins = np.zeros(BINS,dtype='int32')
  avg_force = np.zeros(BINS)
  free_en_endbin = np.zeros(BINS+1)
  cond_prob = np.full(BINS,1./float(BINS))
  log_prob_num = np.zeros(BINS)
  pos_bin = (np.arange(BINS)+0.5)/float(BINS)
  biased_en = np.zeros(BINS)


  ### define integrator 
  integrate = integrator(lmp,Cell,_x,_v,_f,mass=pot.mass,\
                            gamma=gamma,temperature=TEMP,\
                            seed=(rank+1)*int(np.random.uniform(1000)),\
                            block_rad=block_rad, cosmin_mask=None,dt=dt,\
                            spring=spring)
  _time = time.time()


  biasf = 0.
  for t in range(THERM_STEPS): 
    epsi, depsi = cv_new[icv].evaluate(_x)
    _x,_f = integrate.eABF_brownian(depsi,biasf)
    log_prob_num = -((spring/2)*(epsi-pos_bin)*(epsi-pos_bin)-free_en_midbin)/(TEMP*kB)
    log_prob_num = log_prob_num-np.max(log_prob_num) ##avoid overflow
    prob_den = np.sum(np.exp(log_prob_num))
    cond_prob = np.exp(log_prob_num)/prob_den
    springf = -spring*(epsi-pos_bin)
    biasf = np.sum(springf*cond_prob)
    
  stats_file = os.path.join(dump_folder,"stats2.dat")
  statsf = open(stats_file,'w')

  maxepsichange = 0.
  maxxchange = 0.
  springf = 0.
  biasf = 0.

  if spring_scheduler:
    spring = 0. 

  for t in range(TOT_TIME): 
    oldepsi = epsi.copy()
    oldx = _x.copy()
    
    epsi, depsi = cv_new[icv].evaluate(_x)
    if epsi < 1. and epsi > 0.:
      thisbin = math.floor(epsi*BINS)
      n_bins[thisbin] += 1

    _x,_f = integrate.eABF_brownian(depsi,biasf)

    log_prob_num = -((spring/2)*(epsi-pos_bin)*(epsi-pos_bin)-free_en_midbin)/(TEMP*kB)
    log_prob_num = log_prob_num-np.max(log_prob_num) ##avoid overflow
    prob_den = np.sum(np.exp(log_prob_num))
    cond_prob = np.exp(log_prob_num)/prob_den
    if prob_scheduler:
      cond_prob_w = np.minimum(cond_prob*t/schedule_steps,cond_prob)
    else:
      cond_prob_w = cond_prob
    
    springf = -spring*(epsi-pos_bin)
    avg_force_num += springf*cond_prob_w
    avg_force_den += cond_prob_w
    avg_force = avg_force_num/avg_force_den
    if binsmin: ### don't count avg_force if it has never passed from there
      avg_force = avg_force*(avg_force_den>nbinsmin) 
    for i in range(1,BINS+1):  
      free_en_endbin[i] = free_en_endbin[i-1]+(avg_force[i-1]/float(BINS))
      free_en_midbin[i-1] = (free_en_endbin[i]+free_en_endbin[i-1])/2.

    if spring_scheduler:
      spring = min(spring0*t/schedule_steps,spring0)
    springf = -spring*(epsi-pos_bin)
    biasf = np.sum(springf*cond_prob)
      
    epsichange = epsi-oldepsi
    xchange = np.mean(np.abs(pbc_diff(_x-oldx,Cell)))
    xch_rms = np.sqrt(np.mean(np.square(pbc_diff(_x-oldx,Cell))))

    maxepsichange = max(abs(epsichange),maxepsichange)
    maxxchange = max(abs(xchange),maxxchange)
    
    if t%100000 == 0:
      print('step ',t)
    if t%100 == 0:
      statsf.write(str(t)+' '+str(epsi)+' '+str(epsichange)+' '+str(xchange)+' '+str(xch_rms)+' '+str(biasf)+' '+'\n')#+' '+str(biasf)+' '+
    if t%10000 == 0:
      probf.write('Run '+str(icv)+' step '+str(t)+' pos '+str(epsi)+' \n')
      for i in range(BINS):
        probf.write(str(i)+' '+str(cond_prob[i])+' '+str(avg_force_den[i])+' \n')

    if t%10 == 0:
      t10 = t//10
      struct_file = os.path.join(struct_folder,"struct2_%d" % t10)
      xdiff = _x - X0
      xsave = _x.copy()
      ### this works only for 90° angles cell
      for j in range(_N):
        for k in range(3):
          if xdiff[j,k]>Cell[k,k]/2:
            xsave[j,k]-=Cell[k,k]
          if xdiff[j,k]<-Cell[k,k]/2:
            xsave[j,k]+=Cell[k,k]
      xsave.tofile(struct_file)
      #_f.tofile(forces_file)


  print('maxepsichange',maxepsichange,'maxxchange',maxxchange)

  statsf.close()
  print("end run",icv+1)


  for i in range(BINS):
    print(i,avg_force[i],n_bins[i],avg_force_den[i])

  free_energy_bins = np.zeros(BINS+1)
  fen = 0. 
  for i in range(BINS+1):  
    print(i,free_en_endbin[i])
    enf.write(str(i)+' '+str(free_en_endbin[i])+' \n')
  enf.write('&')

enf.close()
probf.close()
exit()
