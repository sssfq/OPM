from tensorflow.python.ops.numpy_ops.np_math_ops import linspace
import opm_inceptionV4
from DataLoader import DataLoader
import tensorflow as tf
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

signal_size = 1024
nominalsymbolrate = 0.5
label = 'OSNR'
train_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train/'
test_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test/'

model_to_be_restored = opm_inceptionV4.InceptionOSNR(input_shape=(1024,4))
checkpoint = tf.train.Checkpoint(CNNmodel=model_to_be_restored)
checkpoint.restore(tf.train.latest_checkpoint('./tensorboard/recorder/opticnception64-32-16'))

# dataloader = DataLoader(signal_size,nominalsymbolrate,label,train_path,test_path)
# dataloader.testsetload()
# y_pred = model_to_be_restored.predict(np.rollaxis(dataloader.test_data,1,3))
# mae = tf.keras.metrics.MeanAbsoluteError()
# mae.update_state(y_true=dataloader.test_label[0:50],y_pred=y_pred[0:50])
# print(mae.result())
# mae.reset_states()
# mae.update_state(y_true=dataloader.test_label[-50:],y_pred=y_pred[-50:])
# print(mae.result())

# for layer in model_to_be_restored.layers:
#     print(layer.name)

model_to_be_restored.build(input_shape=(None,1024,4))

# for i in range(len(model_to_be_restored.get_layer('inception1').get_weights())):
#     print(model_to_be_restored.get_layer('inception1').get_weights()[i].shape)

conv512 = model_to_be_restored.get_layer('inception1').get_weights()[0]
conv256 = model_to_be_restored.get_layer('inception1').get_weights()[3]
conv128 = model_to_be_restored.get_layer('inception1').get_weights()[6]
plt.figure(1)
for num in range(0,8):
    ax = plt.subplot(4,2,num+1)
    for channel in range(0,4):
        a = conv512[:,channel,num]
        A = abs(fft(a,64))
        plt.plot(np.arange(0,64),A,marker='o')

plt.figure(2)
for num in range(0,16):
    ax = plt.subplot(4,4,num+1)
    for channel in range(0,4):
        a = conv256[:,channel,num]
        A = abs(fft(a,32))
        
        plt.plot(np.arange(0,32),A,marker='o')


plt.figure(3)
for num in range(0,8):
    ax = plt.subplot(4,2,num+1)
    for channel in range(0,4):
        a = conv128[:,channel,num]
        A = abs(fft(a,16))
        plt.plot(np.arange(0,16),A,marker='o')

plt.show()
# w = model_to_be_restored.weights[0]
# print(len(w))

# print(model_to_be_restored.trainable_variables)

# model_to_be_restored.summary()