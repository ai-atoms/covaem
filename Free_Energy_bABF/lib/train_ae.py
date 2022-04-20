# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from autoe_1L_noBN_smooth import OAE
import math

tf.config.set_visible_devices([], 'GPU')


def pbc_diff(v,Cell):
    return v-np.dot(Cell,np.floor(np.dot(np.linalg.inv(Cell),v.T)+0.5)).T # make sure no cross-boundary jumps.


class cv_autoenc:
   def __init__(self,Cell,X0,deltaX,ndata,nat,dump_folder,struct_folder):
      self.Cell = Cell
      self.X0 = X0
      self.deltaX = deltaX
      self.ndata = ndata
      self.nat = nat

      indim= nat*3
      xx = np.empty([ndata,indim])
      
      for i in range(ndata):
          infile = os.path.join(struct_folder,"struct_%d" % i)
          xtemp = (np.fromfile(infile)).reshape(nat,3)
          xdiff = xtemp - X0
          xx[i] = pbc_diff(xdiff,Cell).flatten()

      allx = np.copy(xx)

      np.random.shuffle(xx)
    
      n_split = int(0.99*ndata)
      X_train, X_test = np.split(xx,[n_split])
      
      print('read everything')
      #########################################################################
      ncomp=28
      self.mypca = PCA(n_components=ncomp,iterated_power=25)## iterated_power 15 minimum, 25 identical to 50
      pca_train = self.mypca.fit_transform(X_train) 
      pca_test  = self.mypca.transform(X_test)
      allpca = self.mypca.transform(allx)

      print('pca fve')
      for i in range(ncomp):
          print(i+1,self.mypca.explained_variance_ratio_[i],sum(self.mypca.explained_variance_ratio_[:i+1]))
      ###########################################
      print('PCA done')

      XX_train, XX_test = pca_train, pca_test

      meanx = np.average(XX_train,axis=0)
      inv_var = 1./np.sqrt(tf.reduce_mean(tf.pow(XX_train - meanx,2)))
      self.meanx = meanx
      self.inv_var = inv_var
      print('inv_var',inv_var)
      XX_train, XX_test = inv_var*(XX_train-meanx), inv_var*(XX_test-meanx)

      xn_train = tf.random.shuffle(XX_train)


      BATCH_SIZE= 8000
      BUFFER_SIZE= n_split
      train_data = tf.data.Dataset.from_tensor_slices((xn_train,xn_train)).batch(BATCH_SIZE).shuffle(buffer_size = BUFFER_SIZE)
      test_data = tf.data.Dataset.from_tensor_slices((XX_test,XX_test)).batch(BATCH_SIZE)

      ###############################################
      zdim  = 1
      self.auto_encoder = OAE(input_dim = ncomp, dim = [56,zdim],activ='linear', reg =0.001, batch_size = BATCH_SIZE)
      ########################################################################



      loss_fn = tf.keras.losses.MeanSquaredError()
      self.auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.005), loss = loss_fn)
      self.auto_encoder.fit(train_data,  batch_size = BATCH_SIZE, epochs = 80, validation_data=(XX_test , XX_test), verbose=0)
      self.auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.002), loss = loss_fn)
      self.auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 120, validation_data=(XX_test , XX_test), verbose=0)
      self.auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.001), loss = loss_fn)
      self.auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 100, validation_data=(XX_test , XX_test), verbose=0)
      self.auto_encoder.compile(optimizer = tf.optimizers.Adam(learning_rate=0.0002), loss = loss_fn)
      self.auto_encoder.fit(train_data, batch_size = BATCH_SIZE, epochs = 150, validation_data=(XX_test , XX_test), verbose=0)


      train_mse = self.auto_encoder.evaluate(train_data, batch_size = BATCH_SIZE, verbose=2)
      test_mse = self.auto_encoder.evaluate(test_data, batch_size = BATCH_SIZE, verbose=2)
      
      train_var = tf.reduce_mean(tf.pow(XX_train - tf.reduce_mean(XX_train, axis=0),2))  #these is actually 1 if standard scaler is used
      test_var = tf.reduce_mean(tf.pow(XX_test - tf.reduce_mean(XX_test, axis=0),2))  #but this is not (or not exactly)

      print('train_mse',train_mse,'train_var',train_var.numpy())

      fve_train = 1. - float(train_mse/train_var)
      fve_test = 1. - float(test_mse/test_var)

      print('fve',fve_train,fve_test)

      self.auto_encoder.summary()

      allpos1 = allx[:,0]+allx[:,1]+allx[:,2]
      allcv = self.auto_encoder.encode(inv_var*(allpca-meanx)).numpy()

      start = np.zeros([nat,3])
      startpca = self.mypca.transform(np.expand_dims(start.flatten(),axis=0))
      self.startcv = self.auto_encoder.encode(inv_var*(startpca-meanx)).numpy()
      endpca = self.mypca.transform(np.expand_dims((deltaX).flatten(),axis=0))
      self.endcv = self.auto_encoder.encode(inv_var*(endpca-meanx)).numpy()
      self.deltacv = self.endcv-self.startcv

      allcv = (allcv-self.startcv)/self.deltacv
      
      checkfile = os.path.join(dump_folder,"poscv.dat")
      with open(checkfile,'w') as f:
          for i in range(len(allcv)):
              f.write(str(allpos1[i])+' '+str(allcv[i,0])+'\n')

   def evaluate(self,x):
       xdiff = x - self.X0
       x = pbc_diff(xdiff,self.Cell).flatten()
       thispca = self.mypca.transform(np.expand_dims(x,axis=0))
       g_pca = self.mypca.components_
       inp = self.inv_var*(thispca-self.meanx)
       cv,dcv =  self.auto_encoder.encode.enc_grad(inp)
       cv = (cv[0].numpy()-self.startcv)/self.deltacv
       dcv = self.inv_var*(np.dot(dcv[0].numpy(),g_pca)/self.deltacv).reshape(self.nat,3)
       return cv[0][0], dcv

   def generate(self,Ntest):
       shift = 0.5/(Ntest+1)
       cvtest = np.linspace(0.+shift, 1.-shift, num=Ntest, endpoint=True)
       cvtest = cvtest*self.deltacv+self.startcv
       cvtest = np.reshape(cvtest,(Ntest,1))
       xpcatest =  self.auto_encoder.decode(cvtest)
       xpcatest = (xpcatest/self.inv_var)+self.meanx
       xtest = self.mypca.inverse_transform(xpcatest)      
       xtest = np.reshape(xtest,(Ntest,self.nat,3))
       return xtest

   def autoencode(self,x):
       xdiff = x - self.X0
       x = pbc_diff(xdiff,self.Cell).flatten()
       thispca = self.mypca.transform(np.expand_dims(x,axis=0))
       g_pca = self.mypca.components_
       inp = self.inv_var*(thispca-self.meanx)
       cv,dcv =  self.auto_encoder.encode.enc_grad(inp)
       xpcanew =  self.auto_encoder.decode(cv)
       xpcanew = (xpcanew/self.inv_var)+self.meanx
       xnew = self.mypca.inverse_transform(xpcanew)
       xnew = np.reshape(xnew,(self.nat,3))
       return xnew
