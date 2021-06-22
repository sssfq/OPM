import tensorflow as tf
import numpy as np
# This version of CNN reproduce the work of Tanimura Takahito
# DOI: 10.1109/ECOC.2018.8535225
#      10.1364/OFC.2018.Tu3E.3
#      10.1364/JOCN.11.000A52

# dataset import and preprocessing
from DataLoader import DataLoader
# dataset pre-processing

# CNN model initialization
class opm_cnnModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1_1 = tf.keras.layers.Conv1D(
            filters = 16,
            kernel_size = 10,
            padding = 'same',
            activation = 'relu',
            # data_format = 'channels_first',
            dilation_rate = 1
        )

        self.conv1_2 = tf.keras.layers.Conv1D(
            filters = 32,
            kernel_size = 10,
            padding = 'same',
            activation = 'relu',
            # data_format = 'channels_first'
        )

        self.pool1 = tf.keras.layers.MaxPool1D(
            pool_size = 4,
            strides = 4,
            padding = 'valid',
            # data_format = 'channels_first'
        )

        self.conv2 = tf.keras.layers.Conv1D(
            filters = 64,
            kernel_size = 10,
            padding = 'same',
            activation = 'relu',
            # data_format = 'channels_first'
        )

        self.pool2 = tf.keras.layers.MaxPool1D(
            pool_size = 4,
            strides = 4,
            padding = 'valid',
            # data_format = 'channels_first'
        )

        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=2048,activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(units=1)

    def call (self,inputs):
        # print(inputs)
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
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

model = opm_cnnModel()

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Checkpoint
checkpoint = tf.train.Checkpoint(CNNmodel=model,Adamoptimizer=optimizer)
# Tensorboard
summary_writer = tf.summary.create_file_writer('./tensorboard')
# training
num_batches = int(dataloader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    signal,rolloff = dataloader.get_batch(batch_size)
    signal = np.rollaxis(signal,1,3)
    with tf.GradientTape() as tape:
        y_pred = model(signal)
        loss = tf.keras.losses.mean_squared_error(rolloff,y_pred)   
        loss = tf.reduce_mean(loss)
        # print("signal\n:",signal.shape)
        # print("y_pred\n:",y_pred)
        # print("rolloff\n:",rolloff)
        # print(loss)
        print("batch %d: loss %f" % (batch_index,loss.numpy()))
        with summary_writer.as_default():
            tf.summary.scalar("loss",loss,step=batch_index)
    grads = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
    
        
checkpoint.save('./checkpoint/model.ckpt')
################################

# evaluation
mae = tf.keras.metrics.MeanAbsoluteError()
num_batches = int(dataloader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index*batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(np.rollaxis(dataloader.test_data[start_index: end_index],1,3))
    mae.update_state(y_true=dataloader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % mae.result())

# # restore
# model_to_be_restored = opm_cnnModel()
# checkpoint = tf.train.Checkpoint(CNNmodel=model_to_be_restored)
# checkpoint.restore('./checkpoint/model.ckpt')

