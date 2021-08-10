from logging import basicConfig
from re import S
from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import numpy as np
from tensorflow.python.eager.function import BACKWARD_FUNCTION_ATTRIBUTE_NAME
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.layers.pooling import AveragePooling1D, AveragePooling2D
from tensorflow.python.keras.utils.version_utils import ModelVersionSelector
from DataLoader import DataLoader

class BasicConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.conv = layers.Conv1D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x,training=training)
        x = self.relu(x)
        return x

class LinearConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.conv = layers.Conv1D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x,training=training)
        return x

class Stem(layers.Layer):
    def __init__(self,input_shape):
        super().__init__()
        self.conv1 = Sequential([
            layers.Input(input_shape), # recommend to use the functional layer API via Input, which creates an InputLayer without directly using InputLayer
            BasicConv1D(32,3,strides=2),
            BasicConv1D(32,3),
            BasicConv1D(64,3,padding='same')
        ])
        self.branch1_MaxPool = layers.MaxPool1D(
            pool_size = 3,
            strides = 2
        )
        self.branch1_conv = BasicConv1D(96,3,strides=2)

        self.branch2_1 = Sequential([
            BasicConv1D(64,1,padding='same'),
            BasicConv1D(96,3)
        ])
        self.branch2_2 = Sequential([
            BasicConv1D(64,1,padding='same'),
            BasicConv1D(64,7,padding='same'),
            BasicConv1D(96,3)
        ])

        self.branch3_MaxPool = layers.MaxPool1D(
            pool_size= 3,
            strides = 2
        )
        self.branch3_conv = BasicConv1D(192,3,strides=2)


    
    def call(self, x, training=False):
        x = self.conv1(x,training=training)

        x = [
            self.branch1_conv(x,training=training),
            self.branch1_MaxPool(x)
        ]
        x = tf.concat(x,axis=-1)

        x = [
            self.branch2_1(x,training=training),
            self.branch2_2(x,training=training)
        ]
        x = tf.concat(x,axis=-1)

        x = [
            self.branch3_conv(x,training=training),
            self.branch3_MaxPool(x)
        ]
        x = tf.concat(x,axis=-1)

        return x

class InceptionA(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Sequential([
            layers.AveragePooling1D(pool_size=3,strides=1,padding='same'),
            BasicConv1D(96,1,padding='same')
        ])

        self.branch2 = BasicConv1D(96,1,padding='same')
        
        self.branch3 = Sequential([
            BasicConv1D(64,1,padding='same'),
            BasicConv1D(96,3,padding='same')
        ])

        self.branch4 = Sequential([
            BasicConv1D(64,1,padding='same'),
            BasicConv1D(96,3,padding='same'),
            BasicConv1D(96,3,padding='same')
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training),
            self.branch4(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        
        return x

class ReductionA(layers.Layer):
    def __init__(self,k=192,l=224,m=256,n=384):
        super().__init__()
        self.branch1 = Sequential([
            BasicConv1D(k,1),
            BasicConv1D(l,3,padding='same'),
            BasicConv1D(m,3,strides=2)
        ])

        self.branch2 = BasicConv1D(n,3,strides=2)

        self.branch3 = layers.MaxPool1D(pool_size=3,strides=2,padding='valid')

    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x)
        ]
        x = tf.concat(x,axis=-1)
        return x

class InceptionB(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Sequential([
            layers.AveragePooling1D(pool_size=3,strides=1,padding='same'),
            BasicConv1D(128,1,padding='same')
        ])
        
        self.branch2 = BasicConv1D(384,1,padding='same')

        self.branch3 = Sequential([
            BasicConv1D(192,1,padding='same'),
            BasicConv1D(256,7,padding='same')
        ])

        self.branch4 = Sequential([
            BasicConv1D(192,1,padding='same'),
            BasicConv1D(224,7,padding='same'),
            BasicConv1D(256,1,padding='same')
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training),
            self.branch4(x,training=training)
        ]

        x = tf.concat(x,axis=-1)
        return x

class ReductionB(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Sequential([
            BasicConv1D(192,1,padding='same'),
            BasicConv1D(192,3,strides=2)
        ])

        self.branch2 = Sequential([
            BasicConv1D(256,1,padding='same'),
            BasicConv1D(576,7,padding='same'),
            BasicConv1D(320,3,strides=2)
        ])

        self.branch3 = layers.MaxPool1D(pool_size=3,strides=2,padding='valid')
    
    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x)
        ]

        x = tf.concat(x,axis=-1)
        return x

class InceptionC(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Sequential([
            layers.AveragePooling1D(pool_size=3,strides=1,padding='same'),
            BasicConv1D(256,1,padding='same')
        ])

        self.branch2 = BasicConv1D(256,1,padding='same')

        self.branch3 = Sequential([
            BasicConv1D(384,1,padding='same'),
            BasicConv1D(256,3,padding='same')
        ])

        self.branch4 = Sequential([
            BasicConv1D(384,1,padding='same'),
            BasicConv1D(512,3,padding='same'),
            BasicConv1D(256,3,padding='same')
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training),
            self.branch4(x,training=training)
        ]

        x = tf.concat(x,axis=-1)
        return x

class InceptionV4(Model):
    def __init__(self,
                input_shape,
                A=4,B=7,C=3,
                k=192,l=224,m=256,n=384
                ):
        super().__init__()
        self.stem = Stem(input_shape)
        self.inception_a = self._generate_inception_module(A, InceptionA)
        self.reduction_a = ReductionA(k,l,m,n)
        self.inception_b = self._generate_inception_module(B, InceptionB)
        self.reduction_b = ReductionB()
        self.inception_c = self._generate_inception_module(C, InceptionC)

        self.averagepool = layers.AveragePooling1D(pool_size=30)
        self.dropout = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer2 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer3 = layers.Dense(units=1)

    def call(self, input, training=False):
        x = self.stem(input, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.layer1(x)
        # x = self.batchnorm3(x,training=training)
        # x = self.dropout1(x,training=training)
        x = self.layer2(x)
        # x = self.batchnorm4(x,training=training)
        # x = self.dropout2(x,training=training)
        x = self.layer3(x)

        return x

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

class InceptionV4_simplified(Model):
    def __init__(self,
                input_shape,
                k=192,l=224,m=256,n=384
                ):
        super().__init__()
        self.stem = Stem(input_shape)
        self.inception_a = InceptionA()
        self.reduction_a = ReductionA(k,l,m,n)
        self.inception_b = InceptionB()
        self.reduction_b = ReductionB()
        self.inception_c = InceptionC()

        self.averagepool = layers.AveragePooling1D(pool_size=30)
        self.dropout = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer2 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer3 = layers.Dense(units=1)

    def call(self, input, training=False):
        x = self.stem(input, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class InceptionResNetA(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(32,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(32,1,padding='same'),
            BasicConv1D(32,3,padding='same')
        ])
        self.branch3 = Sequential([
            BasicConv1D(32,1,padding="same"),
            BasicConv1D(48,3,padding="same"),
            BasicConv1D(64,3,padding='same')
        ])

        self.inception = BasicConv1D(384,1,padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training),
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual)
        output = self.relu(output)

        return output

class InceptionResNetReductionA(ReductionA):
    def __init__(self):
        super().__init__()

class InceptionResNetB(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(192,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(128,1,padding='same'),
            BasicConv1D(192,7,padding='same')
        ])
        self.inception = BasicConv1D(1024,1,padding='same')

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training)
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual)
        output = self.relu(output)

        return output

class InceptionResNetReductionB(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = layers.MaxPool1D(pool_size=3,strides=2,padding='valid')

        self.branch2 = Sequential([
            BasicConv1D(256,1,padding='same'),
            BasicConv1D(288,3,strides=2)
        ])

        self.branch3 = Sequential([
            BasicConv1D(256,1,padding='same'),
            BasicConv1D(384,3,strides=2)
        ])

        self.branch4 = Sequential([
            BasicConv1D(256,1,padding='same'),
            BasicConv1D(288,3,padding='same'),
            BasicConv1D(320,3,strides=2)
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training),
            self.branch4(x,training=training)
        ]

        x = tf.concat(x,axis=-1)
        return x

class InceptionResNetC(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(192,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(192,1,padding='same'),
            BasicConv1D(256,3,padding='same')
        ])
        self.inception = BasicConv1D(1248,1,padding='same')

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training)
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual)
        output = self.relu(output)

        return output

class InceptionResNetV2(Model):
    def __init__(self,
                input_shape,
                A=5,B=10,C=5,
                k=192,l=224,m=256,n=384
                ):
        super().__init__()
        self.stem = Stem(input_shape)
        self.inception_resnet_a = self._generate_inception_module(A, InceptionResNetA)
        self.reduction_a = ReductionA(k,l,m,n)
        self.inception_resnet_b = self._generate_inception_module(B, InceptionResNetB)
        self.reduction_b = ReductionB()
        self.inception_resnet_c = self._generate_inception_module(C, InceptionResNetC)

        self.averagepool = layers.AveragePooling1D(pool_size=30)
        self.dropout = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer2 = layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer3 = layers.Dense(units=1)

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.inception_resnet_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_resnet_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_resnet_c(x, training=training)

        x = self.averagepool(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class Inception_FIR(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = LinearConv1D(8,512,padding='same')

        self.branch2 = LinearConv1D(16,256,padding='same')
        
        self.branch3 = Sequential([
            # LinearConv1D(32,1,padding='same'),
            LinearConv1D(8,128,padding='same')
        ])


    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        
        return x

class Inception_miniFIR(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = LinearConv1D(8,64,padding='same')

        self.branch2 = LinearConv1D(16,32,padding='same')
        
        self.branch3 = Sequential([
            # LinearConv1D(32,1,padding='same'),
            LinearConv1D(8,16,padding='same')
        ])


    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        
        return x

class Inception1_avgpool(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Sequential([
            LinearConv1D(8,512,padding='same'),
            AveragePooling1D(pool_size=8,strides=1,padding='same')])

        self.branch2 = Sequential([
            LinearConv1D(16,256,padding='same'),
            AveragePooling1D(pool_size=8,strides=1,padding='same')])
        
        self.branch3 = Sequential([
            # LinearConv1D(32,1,padding='same'),
            LinearConv1D(8,128,padding='same'),
            AveragePooling1D(pool_size=8,strides=1,padding='same')])


    def call(self, x, training=False):
        x = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        
        return x

class InceptionResNet1(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(32,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(32,1,padding='same'),
            BasicConv1D(32,16,padding='same')
        ])
        self.branch3 = Sequential([
            BasicConv1D(32,1,padding='same'),
            BasicConv1D(32,64,padding='same'),
            BasicConv1D(32,32,padding='same')
        ])
        self.inception = LinearConv1D(32,3,padding='same')

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual*0.1)
        output = self.relu(output)

        return output

class Reduction1(layers.Layer):
    def __init__(self,pool_size=513):
        super().__init__()
        self.branch1 = AveragePooling1D(pool_size=pool_size,strides=1)

        self.branch2 = BasicConv1D(32,pool_size)

        self.branch3 = Sequential([
            BasicConv1D(64,1),
            BasicConv1D(16,pool_size)
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        return x

class InceptionResNet2(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(64,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(32,1,padding='same'),
            BasicConv1D(64,3,padding='same')
        ])
        self.branch3 = Sequential([
            BasicConv1D(32,1,padding='same'),
            BasicConv1D(48,5,padding='same'),
            BasicConv1D(64,3,padding='same')
        ])
        self.inception = LinearConv1D(80,3,padding='same')

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual*0.1)
        output = self.relu(output)

        return output

class Reduction2(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = AveragePooling1D(pool_size=8,strides=8)

        self.branch2 = BasicConv1D(64,8,strides=8)

        self.branch3 = Sequential([
            BasicConv1D(32,1),
            BasicConv1D(64,8,strides=8)
        ])

    def call(self, x, training=False):
        x = [
            self.branch1(x),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        x = tf.concat(x,axis=-1)
        return x

class InceptionResNet3(layers.Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = BasicConv1D(120,1,padding='same')
        self.branch2 = Sequential([
            BasicConv1D(128,1,padding='same'),
            BasicConv1D(72,3,padding='same')
        ])
        self.branch3 = Sequential([
            AveragePooling1D(pool_size=5,padding='same',strides=1),
            BasicConv1D(64,1,padding='same')
        ])
        self.inception = LinearConv1D(208,64,padding='same')

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,x,training=False):
        residual = [
            self.branch1(x,training=training),
            self.branch2(x,training=training),
            self.branch3(x,training=training)
        ]
        residual = tf.concat(residual,axis=-1)
        residual = self.inception(residual,training=training)

        shortcut = x
        output = self.bn(shortcut+residual*0.1)
        output = self.relu(output)

        return output

class OptInception(Model):
    def __init__(self,input_shape):
        super().__init__()
        self.inputlayer = Sequential([layers.InputLayer(input_shape)])
        self.inception_1 = Inception_FIR()
        self.inceptionres_1 = self._generate_inception_module(1,InceptionResNet1)
        self.reduction_1 = Reduction1()
        self.inceptionres_2 = self._generate_inception_module(1,InceptionResNet2)
        self.reduction_2 = Reduction2()
        self.inceptionres_3 = self._generate_inception_module(1,InceptionResNet3)

        self.averagepool = layers.AveragePooling1D(pool_size=16,strides=16)
   
        self.flatten = layers.Flatten()
        self.bn = layers.BatchNormalization()
        self.layer1 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.4)
        self.layer2 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.4)
        # self.layer3 = layers.Dense(units=416,activation=tf.nn.leaky_relu)
        self.layer3 = layers.Dense(units=1)
        
    @tf.function
    def call(self, input, training=False):
        x = self.inputlayer(input)
        x = self.inception_1(x, training=training)
        x = self.inceptionres_1(x, training=training)
        x = self.reduction_1(x, training=training)
        x = self.inceptionres_2(x, training=training)
        x = self.reduction_2(x, training=training)
        x = self.inceptionres_3(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        # x = self.dropout2(x, training=training)
        # x = self.bn(x,training=training)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        # x = self.layer4(x)

        return x

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

class OptInception_miniFIR(Model):
    def __init__(self,input_shape):
        super().__init__()
        self.inputlayer = Sequential([layers.Input(input_shape)])
        self.inception_1 = Inception_miniFIR()
        self.inceptionres_1 = self._generate_inception_module(1,InceptionResNet1)
        self.reduction_1 = Reduction1()
        self.inceptionres_2 = self._generate_inception_module(1,InceptionResNet2)
        self.reduction_2 = Reduction2()
        self.inceptionres_3 = self._generate_inception_module(1,InceptionResNet3)

        self.averagepool = layers.AveragePooling1D(pool_size=16,strides=16)
   
        self.flatten = layers.Flatten()
        self.bn = layers.BatchNormalization()
        self.layer1 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.4)
        self.layer2 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.4)
        # self.layer3 = layers.Dense(units=416,activation=tf.nn.leaky_relu)
        self.layer3 = layers.Dense(units=1)
        
    @tf.function
    def call(self, input, training=False):
        x = self.inputlayer(input)
        x = self.inception_1(x, training=training)
        x = self.inceptionres_1(x, training=training)
        x = self.reduction_1(x, training=training)
        x = self.inceptionres_2(x, training=training)
        x = self.reduction_2(x, training=training)
        x = self.inceptionres_3(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        # x = self.dropout2(x, training=training)
        # x = self.bn(x,training=training)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        # x = self.layer4(x)

        return x

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

class OptInception_Pro(Model):
    def __init__(self,input_shape):
        super().__init__()
        self.inputlayer = Sequential([layers.InputLayer(input_shape)])
        self.inception_1 = Inception_FIR()
        self.inceptionres_1 = self._generate_inception_module(5,InceptionResNet1)
        self.reduction_1 = Reduction1()
        self.inceptionres_2 = self._generate_inception_module(5,InceptionResNet2)
        self.reduction_2 = Reduction2()
        self.inceptionres_3 = self._generate_inception_module(3,InceptionResNet3)

        self.averagepool = layers.AveragePooling1D(pool_size=16,strides=16)
   
        self.flatten = layers.Flatten()
        self.bn = layers.BatchNormalization()
        self.layer1 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.4)
        self.layer2 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.4)
        # self.layer3 = layers.Dense(units=416,activation=tf.nn.leaky_relu)
        self.layer3 = layers.Dense(units=1)
        
    @tf.function
    def call(self, input, training=False):
        x = self.inputlayer(input)
        x = self.inception_1(x, training=training)
        x = self.inceptionres_1(x, training=training)
        x = self.reduction_1(x, training=training)
        x = self.inceptionres_2(x, training=training)
        x = self.reduction_2(x, training=training)
        x = self.inceptionres_3(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        # x = self.dropout2(x, training=training)
        # x = self.bn(x,training=training)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        # x = self.layer4(x)

        return x

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

class OptInception_chromdisp(Model):
    def __init__(self,input_shape):
        super().__init__()
        self.inputlayer = Sequential([layers.InputLayer(input_shape)])
        self.inception_1 = Inception_FIR()
        self.inceptionres_1 = self._generate_inception_module(1,InceptionResNet1)
        self.reduction_1 = Reduction1()
        self.inceptionres_2 = self._generate_inception_module(1,InceptionResNet2)
        self.reduction_2 = Reduction2()
        self.inceptionres_3 = self._generate_inception_module(1,InceptionResNet3)

        self.averagepool = layers.AveragePooling1D(pool_size=16,strides=16)
   
        self.flatten = layers.Flatten()
        self.bn = layers.BatchNormalization()
        self.layer1 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.4)
        self.layer2 = layers.Dense(units=832,activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.4)
        # self.layer3 = layers.Dense(units=416,activation=tf.nn.leaky_relu)
        self.layer3 = layers.Dense(units=1)
        
    @tf.function
    def call(self, input, training=False):
        x = self.inputlayer(input)
        x = self.inception_1(x, training=training)
        x = self.inceptionres_1(x, training=training)
        x = self.reduction_1(x, training=training)
        x = self.inceptionres_2(x, training=training)
        x = self.reduction_2(x, training=training)
        x = self.inceptionres_3(x, training=training)
        
        x = self.averagepool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        # x = self.dropout2(x, training=training)
        # x = self.bn(x,training=training)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)

        # x = self.layer4(x)

        return x

    @staticmethod
    def _generate_inception_module(block_num, block):
        nets = Sequential()
        for num in range(block_num):
            nets.add(block())
        return nets

if __name__ == '__main__':
    signal_size = 1024
    # model1 = InceptionV4(input_shape=(signal_size,4))
    # print('InceptionV4 Instantiation Success!')

    # model2 = InceptionV4_simplified(input_shape=(signal_size,4))
    # print('InceptionV4_simplified Instantiation Success!')

    # model3 = InceptionResNetV2(input_shape=(signal_size,4))
    # print('InceptionResNetV2 Instantiation Success!')

    model4 = InceptionOSNR(input_shape=(signal_size,4))
    print('InceptionOSNR Instantiation Success!')

    signal = np.random.randn(5,signal_size,4)
    y_pred = model4.predict(signal)
    print(y_pred.shape)
    y_call = model4(signal)
    print(y_call.shape)
    tf.saved_model.save(model4, "tensorboard/modelsave/1")



