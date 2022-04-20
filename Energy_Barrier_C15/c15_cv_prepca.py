# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from autoe_test_noBN import OAE
import math
from scipy.interpolate import interp1d, CubicSpline

tf.config.set_visible_devices([], 'GPU')

def  get_random(deltax, size):
    mean = [0,0,0]
    covar = np.array([[1.0,0.0,0], [0,1,0],[0,0,1]], dtype='float')*deltax
    x = np.random.multivariate_normal(mean=[0,0,0], cov=covar, size=size)
    ###make sure that is has 0 mean so that there is no translation
    x = x - np.mean(x,axis=0)
    return x;

ndata = 50000
nat = 1026
indim = nat*3 

all_data = []
for i in range(24):
    my_data = []
    infile = "path/knot_"+str(i)+".xyz"
    with open(infile) as f:
        lines_list = f.readlines()
        for line in lines_list:
            for val in line.split():
                my_data.append(float(val))
        f.close()
    all_data.append(my_data)

alldata = np.array(all_data)

x = np.linspace(0, 1, num=24, endpoint=True)
#‘not-a-knot’ (default): The first and second segment at a curve end are the same polynomial
#f = CubicSpline(x,alldata, axis = 0,bc_type='not-a-knot')
#‘natural’: The second derivative at curve ends are zero
f = CubicSpline(x,alldata, axis = 0,bc_type='natural')
df = f.derivative()

xnew = np.linspace(-0.05, 1.05, num=ndata, endpoint=True)
np.random.shuffle(xnew)

ynew = f(xnew)
dynew = df(xnew)

xx = np.empty([ndata,indim])

for j in range(ndata):
    pos_cv = ynew[j]
    der_cv = dynew[j]
    tmp=get_random(0.005,nat)
    noise = tmp.flatten()
    noise = noise - np.dot(noise,der_cv)*der_cv/np.dot(der_cv,der_cv)
    xx[j]=pos_cv+noise

    
n_split = int(0.999*ndata)
X_train, X_test = np.split(xx,[n_split])
pos1_train, pos1_test = np.split(xnew,[n_split])

all_data = None
alldata = None
ynew = None
dynew = None
f = None
df = None
xx = None

print('generated data')
#########################################################################
ncomp=10
mypca = PCA(n_components=ncomp,iterated_power=25)
pca_train = mypca.fit_transform(X_train) 
pca_test  = mypca.transform(X_test)

print('pca fve')
for i in range(ncomp):
    print(i+1,mypca.explained_variance_ratio_[i],sum(mypca.explained_variance_ratio_[:i+1]))
###########################################
print('PCA done')

XX_train, XX_test = pca_train, pca_test

meanx = np.average(XX_train,axis=0)
inv_var = 10./np.sqrt(tf.reduce_mean(tf.pow(XX_train - meanx,2)))
print('inv_var',inv_var)
XX_train, XX_test = inv_var*(XX_train-meanx), inv_var*(XX_test-meanx)

xn_train = tf.random.shuffle(XX_train)


BATCH_SIZE= 1000
BUFFER_SIZE= n_split
train_data = tf.data.Dataset.from_tensor_slices((xn_train,xn_train)).batch(BATCH_SIZE).shuffle(buffer_size = BUFFER_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((XX_test,XX_test)).batch(BATCH_SIZE)

###############################################
zdim  = 1
auto_encoder = OAE(input_dim = ncomp, dim = [10,10,zdim],activ='relu', reg =0.0001, batch_size = BATCH_SIZE)#,z_activ= 'linear'
########################################################################

loss_fn = tf.keras.losses.MeanSquaredError()
#loss_fn = custom_loss()
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.02), loss = loss_fn)
auto_encoder.fit(train_data,  batch_size = BATCH_SIZE, epochs = 20, validation_data=(XX_test , XX_test))
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.005), loss = loss_fn)
auto_encoder.fit(train_data,  batch_size = BATCH_SIZE, epochs = 80, validation_data=(XX_test , XX_test))
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.001), loss = loss_fn)
auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 100, validation_data=(XX_test , XX_test))
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.0001), loss = loss_fn)
auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 120, validation_data=(XX_test , XX_test))
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.00005), loss = loss_fn)
auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 150, validation_data=(XX_test , XX_test))
auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.000005), loss = loss_fn)
auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 200, validation_data=(XX_test , XX_test))

train_mse = auto_encoder.evaluate(train_data, batch_size = BATCH_SIZE)
test_mse = auto_encoder.evaluate(test_data, batch_size = BATCH_SIZE)

train_var = tf.reduce_mean(tf.pow(XX_train - tf.reduce_mean(XX_train, axis=0),2))  #these is actually 1 if standard scaler is used
test_var = tf.reduce_mean(tf.pow(XX_test - tf.reduce_mean(XX_test, axis=0),2))  #buth this is not (or not exactly)

print('train_mse',train_mse,'train_var',train_var.numpy())

fve_train = 1. - float(train_mse/train_var)
fve_test = 1. - float(test_mse/test_var)

print('fve',fve_train,fve_test)

auto_encoder.summary()

d_train = auto_encoder.encode(XX_train).numpy()

### OUTPUT FILE: position on interpolated NEB path vs collective variable from AE bottleneck for all database
with open('colvar.dat',"w") as f2:
    for i in range(len(d_train)):
        f2.write(str(pos1_train[i])+" "+str(d_train[i,0])+"\n")

############################################################

numneb = 24

data = []
for i in range(numneb):
    infile = "path/knot_"+str(i)+".xyz"
    with open(infile) as f:
        readfile = f.readlines()
        x = [[float(val) for val in line.split()] for line in readfile]
        #x = [float(val) for val in readfile[0].split()]
        f.close()
        data.append(x)

x = np.array(data)
x = np.reshape( x, (x.shape[0], x.shape[1] * x.shape[2]))

###########################################
pc_neb = mypca.transform(x)
g_pca = mypca.components_
gpc_neb = np.array([[[g_pca[k,i] for i in range(indim)] for j in range(numneb)] for k in range(ncomp)])
############################################

d_neb, g_neb = auto_encoder.encode.enc_grad(inv_var*(pc_neb-meanx))

d_neb = np.array(d_neb)
g_neb = inv_var*np.array(g_neb)

metr =  np.empty([numneb,zdim,zdim])
metr_pca = np.empty([numneb,ncomp,ncomp])
metr 
for i in range(numneb):
    g1 = g_neb[:,i,:]
    gg = np.matmul(g1,tf.transpose(g1))
    gg = gg #+ 1.e-9*np.eye(zdim)
    metr[i] = np.linalg.inv(gg)
    g1 = gpc_neb[:,i,:]
    gg = np.matmul(g1,tf.transpose(g1))
    gg = gg #+ 1.e-9*np.eye(ncomp)    
    metr_pca[i] = np.linalg.inv(gg)


data = []
for i in range(numneb):
    infile = "path/snapneb."+str(i)+".lmc"
    with open(infile) as f:
        readfile = f.readlines()
        theslice = [5,6,7]
        fx = [[line.split()[j] for j in theslice] for line in readfile[9:2763]]
        f.close()
        data.append(fx)

fx = np.array(data,dtype=np.float32)
fx = np.reshape( fx, (fx.shape[0], fx.shape[1] * fx.shape[2]))

###########################


fcv0_pca = np.array([[np.dot(gpc_neb[j,i,:],fx[i,:]) for i in range(fx.shape[0])] for j in range(gpc_neb.shape[0])])

fcv_pca = np.empty_like(fcv0_pca)
for i in range(numneb):
    fcv_pca[:,i] = np.matmul(fcv0_pca[:,i],metr_pca[i,:,:])

fcv0 = np.array([[np.dot(g_neb[j,i,:],fcv_pca[:,i]) for i in range(fx.shape[0])] for j in range(g_neb.shape[0])])

fcv = np.empty_like(fcv0)
for i in range(numneb):
    fcv[:,i] = np.matmul(fcv0[:,i],metr[i,:,:])

e_test = [0.]
e_ae   = [0.]
e_pca  = [0.]
ene = 0.
eneae = 0.
enepca = 0.

for i in range(1,fx.shape[0]):
    ene = ene - np.dot((x[i]-x[i-1]),(fx[i]+fx[i-1])/2.)
    eneae = eneae - np.dot((d_neb[i,:]-d_neb[i-1,:]),(fcv[:,i]+fcv[:,i-1]))/2.
    enepca = enepca - np.dot((pc_neb[i,:]-pc_neb[i-1,:]),(fcv_pca[:,i]+fcv_pca[:,i-1]))/2.
    e_test.append(ene)
    e_ae.append(eneae)
    e_pca.append(enepca)

print('NEB pos    CV    Eneb    Eae+pca   Epca',)
for i in range(int(fx.shape[0])):
        print(i,d_neb[i][0],e_test[i],e_ae[i],e_pca[i])

### OUTPUT FILE: reference energy profile (integrated on full 3N forces and displacements), AE+PCA energy profile (integrated on reaction coordinate), PCA energy profile (integrated on 10 principal components) 
with open('Eprofile.dat',"w") as f2:
    for i in range(int(fx.shape[0])):
        f2.write(str(i)+" "+str(e_ae[i])+" "+str(e_test[i])+" "+str(e_pca[i])+"\n")
        


