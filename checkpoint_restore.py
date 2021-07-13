import opm_inceptionV4
from DataLoader import DataLoader
import tensorflow as tf
import numpy as np

signal_size = 1024
nominalsymbolrate = 0.5
label = 'OSNR'
train_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train_data/'
test_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test_data/'

model_to_be_restored = opm_inceptionV4.InceptionV4(input_shape=(1024,4))
checkpoint = tf.train.Checkpoint(CNNmodel=model_to_be_restored)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoint'))

dataloader = DataLoader(signal_size,nominalsymbolrate,label,train_path,test_path)
dataloader.testsetload()
y_pred = model_to_be_restored.predict(np.rollaxis(dataloader.test_data,1,3))
mae = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(y_true=dataloader.test_label[0:50],y_pred=y_pred[0:50])
print(mae.result())
mae.reset_states()
mae.update_state(y_true=dataloader.test_label[-50:],y_pred=y_pred[-50:])
print(mae.result())