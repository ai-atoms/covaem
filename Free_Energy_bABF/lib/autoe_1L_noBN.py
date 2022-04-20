from scipy.linalg import orth
import tensorflow as tf
import numpy as np

class Dense_transp(tf.keras.layers.Layer):
    def __init__(self,dense,layer_size,activation=None,**kwargs):
        self.dense = dense
        self.layer_size = layer_size
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self,batch_input_shape):
        self.biases = self.add_weight(name="bias",shape=(self.layer_size,),initializer="zeros")  
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs,self.dense.weights[0],transpose_b=True)
        return self.activation( z + self.biases )
    
class OAE(tf.keras.Model):
    def __init__(self, input_dim=55, dim = [40, 1], batch_size = 128 ,activ= 'relu',reg=0.00001,name='OAE',**kwargs):
        super(OAE, self).__init__(name = name, **kwargs)
        self.input_dim = input_dim
        self.dim_layer = dim
        self.batch_size = batch_size
        self.size_bottle = int(dim[1])
        self.encode_l = [input_dim] + dim[:2]
        self.activ = activ

        self.encode = OAE.Encoder(input_dim,dim,activ,reg)
        self.decode = OAE.Decoder(self.encode,input_dim,dim,activ)


    @tf.function
    def call(self, inputs):
        xenc = self.encode(inputs)
        xdec = self.decode(xenc)
        return xdec

        ############ encode #################################

    class Encoder(tf.keras.layers.Layer):
        def __init__(self,input_dim,dim,activ,reg,name ='encoder',**kwargs):
             super().__init__(name = name, **kwargs)
            self.input_dim = input_dim
            self.dim=dim
            self.activ=activ
 

            initial_k =tf.keras.initializers.GlorotNormal()
            initial_k2 =tf.keras.initializers.GlorotNormal()
            regular = tf.keras.regularizers.l2(reg)


            self.encode_layer1 = tf.keras.layers.Dense(self.dim[0], input_shape = (input_dim,), kernel_initializer=initial_k, bias_initializer=tf.keras.initializers.Zeros() ,activation =activ,name="encode_hidden",kernel_regularizer=regular)
            self.encode_layer2= tf.keras.layers.Dense(self.dim[1], input_shape = (dim[0],), kernel_initializer=initial_k, bias_initializer=tf.keras.initializers.Zeros(),name="bottle", activation ='linear')

        @tf.function
        def call(self, inputs):
            """
            xini = self.flatten_layer(inputs)
            """
            x    = self.encode_layer1 (inputs)
            xcod    = self.encode_layer2 (x)
            return xcod

        @tf.function
        def enc_grad(self, inputs):
            with tf.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch(inputs)
                x    = self.encode_layer1 (inputs)
                xcod    = self.encode_layer2 (x)
            g1 = t1.gradient(xcod, inputs)
            return xcod, g1
        

        def get_layer1(self):
            return self.encode_layer1
        def get_layer2(self):
            return self.encode_layer2
    ########################################################################

    ####### decode #########################################################

    class Decoder(tf.keras.layers.Layer):

        def __init__(self,encoder,input_dim,dim,activ,name ='decoder',**kwargs):
            super().__init__(name = name, **kwargs)
            self.encoder = encoder
            self.input_dim = input_dim
            self.dim=dim
            
            self.decode_layer1 = Dense_transp(self.encoder.get_layer2(),dim[0],activation=activ,name="decode_hidden")
            self.decode_layer2 = Dense_transp(self.encoder.get_layer1(),input_dim,activation='linear',name="output_layer")

        @tf.function
        def call(self, inputs):
            x    = self.decode_layer1 (inputs)
            xrec    = self.decode_layer2 (x)
            return xrec

        
