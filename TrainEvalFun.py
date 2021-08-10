import tensorflow as tf
import numpy as np
from DataLoader import DataLoader
import opm_inceptionV4

def trainevalfun(
    batch_size=50,learning_rate=0.0001,
    signal_size=1024,nominalsymbolrate=0.5,
    num_epochs=10,label='OSNR',
    train_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train_data/',
    test_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test_data/'
    ):
    batch_size = int(batch_size)
    signal_size = int(signal_size)
    num_epochs = int(num_epochs)
    # Data load
    dataloader = DataLoader(signal_size,nominalsymbolrate,label,train_path,test_path)
    dataloader()
    scale=1
    if label == 'ChromaticDispersion':
        scale=15
        dataloader.train_label=dataloader.train_label/scale
        dataloader.test_label=dataloader.test_label/scale
    # Instantiate model
    model = opm_inceptionV4.InceptionOSNR(input_shape=(signal_size,4))
    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Checkpoint
    checkpoint = tf.train.Checkpoint(CNNmodel=model,Adamoptimizer=optimizer)
    # Tensorboard
    summary_writer = tf.summary.create_file_writer('./tensorboard')

    # Training
    num_batches = int(dataloader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        signal,osnr = dataloader.get_batch(batch_size)
        signal = np.rollaxis(signal,1,3) # signal (batch_size,signal_size,channels(4))
        with tf.GradientTape() as tape:
            y_pred = model(signal,training=True)
            loss = tf.keras.losses.mean_squared_error(osnr,y_pred)   
            loss = tf.reduce_mean(loss)
            # print("signal\n:",signal.shape)
            # print("y_pred\n:",y_pred)
            # print("rolloff\n:",rolloff)
            # print(loss)
            print("batch %d: loss %f" % (batch_index,loss.numpy()))
            with summary_writer.as_default():
                tf.summary.scalar("loss",loss,step=batch_index)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,model.trainable_variables))
    
    checkpoint.save('./checkpoint/model'+str(batch_size)+'_'+str(learning_rate)+'.ckpt')
    # Evaluation
    mae = tf.keras.metrics.MeanAbsoluteError()
    num_batches_test = int(dataloader.num_test_data // batch_size)
    for batch_index_test in range(num_batches_test):
        start_index, end_index = batch_index_test*batch_size, (batch_index_test + 1) * batch_size
        y_pred = model.predict(np.rollaxis(dataloader.test_data[start_index: end_index],1,3))
        mae.update_state(y_true=dataloader.test_label[start_index: end_index], y_pred=y_pred)
    print("Mean absolute error of "+label+" estimation:",mae.result())
    return -mae.result()

if __name__ == '__main__':
    trainevalfun(batch_size=40,num_epochs=6,learning_rate=0.0005)