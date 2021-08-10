import tensorflow as tf
import numpy as np
from DataLoader import DataLoader
import opm_inceptionV4
import pandas as pd
def trainevalfun(
    batch_size=50,learning_rate=0.0001,
    signal_size=1024,nominalsymbolrate=0.5,
    num_epochs=10,label='OSNR',
    train_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train_data/',
    test_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test_data/',
    model_save_file='default',
    checkpoint_path='./checkpoint/model.ckpt',
    model = opm_inceptionV4.OptInception(input_shape=(1024,4))
    ):
    batch_size = int(batch_size)
    signal_size = int(signal_size)
    num_epochs = int(num_epochs)
    # Data load
    dataloader = DataLoader(signal_size,nominalsymbolrate,label,train_path,test_path)
    dataloader()
    scale = 1
    if label == 'ChromaticDispersion':
        scale = 1e5
        dataloader.train_label=dataloader.train_label/scale
        dataloader.test_label=dataloader.test_label/scale
    # Instantiate model
    model = model
    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Checkpoint
    checkpoint = tf.train.Checkpoint(CNNmodel=model,Adamoptimizer=optimizer)
    # Tensorboard
    summary_writer = tf.summary.create_file_writer('./tensorboard')
    # Metrics
    mae = tf.keras.metrics.MeanAbsoluteError()
    maeresult = []

    rmse = tf.keras.metrics.RootMeanSquaredError()
    rmseresult = []
    # Training
    num_batches = int(dataloader.num_train_data // batch_size * num_epochs)
    epoch = 1
    # print((num_batches/num_epochs*np.arange(1,num_epochs+1)-1).astype(np.int32),'\n',num_batches)
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
        # Evaluate in each end of epoch
        if batch_index in (num_batches/num_epochs*np.arange(1,num_epochs+1)-1).astype(np.int32):
            # metrics--mae
            mae.reset_states()
            num_batches_test = int(dataloader.num_test_data // batch_size)
            for batch_index_test in range(num_batches_test):
                start_index, end_index = batch_index_test*batch_size, (batch_index_test + 1) * batch_size
                y_pred = model.predict(np.rollaxis(dataloader.test_data[start_index: end_index],1,3))
                mae.update_state(y_true=scale*dataloader.test_label[start_index: end_index], y_pred=scale*y_pred)
            print("Mean absolute error of OSNR estimation in epoch %d:" % epoch ,mae.result())
            maeresult.append(mae.result())
            # metric--mse
            rmse.reset_states()
            num_batches_test = int(dataloader.num_test_data // batch_size)
            for batch_index_test in range(num_batches_test):
                start_index, end_index = batch_index_test*batch_size, (batch_index_test + 1) * batch_size
                y_pred = model.predict(np.rollaxis(dataloader.test_data[start_index: end_index],1,3))
                rmse.update_state(y_true=scale*dataloader.test_label[start_index: end_index], y_pred=scale*y_pred)
            print("Root mean square error of OSNR estimation in epoch %d:" % epoch ,rmse.result())
            rmseresult.append(rmse.result())
            epoch += 1

    tf.saved_model.save(model, "tensorboard/modelsave/"+model_save_file)
    checkpoint.save(checkpoint_path)
    print(maeresult)
    print(rmseresult)
    # return -mae.result() #for bayesian optimization
    return np.array(maeresult),np.array(rmseresult)

if __name__ == '__main__':
    mae,rmse = trainevalfun(batch_size=40,num_epochs=12,learning_rate=0.0005,
                            train_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train/',
                            test_path='F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test/')
    save = pd.DataFrame({'mae':mae,'rmse':rmse})
    save.to_csv('./trainprocess/testresultdata/optinception_minifir.csv',index=False,sep=',')


