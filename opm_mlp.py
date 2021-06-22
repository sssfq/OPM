import tensorflow as tf
import numpy as np
# dataset import and preprocessing
from DataLoader import DataLoader
# dataset pre-processing

# MLP model initialization
class opm_mlpModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(data_format = 'channels_first')
        self.layer1 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(units=1)

    def call (self,inputs):
        x = self.flatten(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output = x
        # output = tf.nn.softmax(x)  #softmax function should be replaced
        return output

# hyperparameter
num_epochs =5
batch_size = 8
learning_rate = 0.001
# Instantiate
signal_size = 512
label = 'Rolloff'
train_path = 'F:/tempo sim data/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6/train_data/'
test_path = 'F:/tempo sim data/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6/test_data/'

dataloader = DataLoader(signal_size,label,train_path,test_path)

model = opm_mlpModel()

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# training
num_batches = int(dataloader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    signal,rolloff = dataloader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(signal)
        loss = tf.keras.losses.mean_squared_error(rolloff,y_pred)   
        loss = tf.reduce_mean(loss)
        # print("signal\n:",signal.shape)
        # print("y_pred\n:",y_pred)
        # print("rolloff\n:",rolloff)
        # print(loss)
        print("batch %d: loss %f" % (batch_index,loss.numpy()))
    grads = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

################################

# evaluation
mae = tf.keras.metrics.MeanAbsoluteError()
num_batches = int(dataloader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index*batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(dataloader.test_data[start_index: end_index])
    mae.update_state(y_true=dataloader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % mae.result())


