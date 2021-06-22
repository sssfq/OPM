import os
import numpy as np
import math

from extract_OptiSystem import extract_OptiSystem
# Combination of data and label from files in folder into an array of num_data*4*signal_size and 1*num_data
class DataLoader():
    def __init__(self,signal_size,label,train_path,test_path):
        self.train_label = []
        self.train_data  = []
        self.num_train_data = 0

        self.test_label = []
        self.test_data  = []
        self.num_test_data = 0
        # read file list
        trainfilelist = os.listdir(train_path)
        testfilelist = os.listdir(test_path)
        # train data (nparray[num_train_data,4,signal_size]) and label (nparray[num_train_data]) extraction
        for filename in trainfilelist:
            filepath = os.path.join(train_path,filename)
            data_loader = extract_OptiSystem(filepath)

            sequencelength = data_loader.params('SequenceLength')
            num_signal_cur_file = math.floor(sequencelength/signal_size)
            train_label_cur_file = data_loader.params(label)
            # calculation the total number of training data with varying sequence lengthes in files
            self.num_train_data += num_signal_cur_file
            # extract training label
            self.train_label.append(np.zeros(num_signal_cur_file)+train_label_cur_file)
            # extract training data
            self.train_data.append(data_loader.signal_channel(signal_size))
        
        self.train_data = np.array(self.train_data).reshape(-1,4,signal_size)
        self.train_label = np.expand_dims(np.array(self.train_label).reshape(-1),axis=-1)

        #test data extraction
        for filename in testfilelist:
            filepath = os.path.join(test_path,filename)
            data_loader = extract_OptiSystem(filepath)

            sequencelength = data_loader.params('SequenceLength')
            num_signal_cur_file = math.floor(sequencelength/signal_size)
            test_label_cur_file = data_loader.params(label)
            # calculation the total number of training data with varying sequence lengthes in files
            self.num_test_data += num_signal_cur_file
            # extract training label
            self.test_label.append(np.zeros(num_signal_cur_file)+test_label_cur_file)
            # extract training data
            self.test_data.append(data_loader.signal_channel(signal_size))
        
        self.test_data = np.array(self.test_data).reshape(-1,4,signal_size)
        self.test_label = np.expand_dims(np.array(self.test_label).reshape(-1),axis=-1)
    
    def get_batch(self,batch_size):
        index = np.random.randint(0,self.num_train_data,batch_size)
        return self.train_data[index,:],self.train_label[index,:]

if __name__ == '__main__':
    signal_size = 512
    label = 'OSNR'
    train_path = 'F:/tempo sim data/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6/train_data/'
    test_path = 'F:/tempo sim data/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6/test_data/'
    dataloader = DataLoader(signal_size,label,train_path,test_path)
    print(dataloader.train_data[64,1,2])
    print(dataloader.train_label.shape)
    print(dataloader.test_data.shape)
    print(dataloader.test_label.shape)
    traindata,trainlabel = dataloader.get_batch(2)
    print(traindata.shape,'\n',trainlabel)