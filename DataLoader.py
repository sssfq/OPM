import os
import numpy as np
import math

from extract_OptiSystem import extract_OptiSystem
# Combination of data and label from files in folder into an array of num_data*4*signal_size and 1*num_data
class DataLoader():
    def __init__(self,signal_size,nominalsymbolrate,label,train_path,test_path):
        self.signal_size = signal_size
        self.nominalsymbolrate = nominalsymbolrate
        self.label = label
        self.train_path = train_path
        self.test_path = test_path

        self.train_label = []
        self.train_data  = []
        self.num_train_data = 0

        self.test_label = []
        self.test_data  = []
        self.num_test_data = 0
        # read file list
        self.trainfilelist = os.listdir(self.train_path)
        self.testfilelist = os.listdir(self.test_path)
        # train data (nparray[num_train_data,4,signal_size]) and label (nparray[num_train_data]) extraction
    def trainsetload(self):
        for filename in self.trainfilelist:
            filepath = os.path.join(self.train_path,filename)
            data_loader = extract_OptiSystem(filepath,self.nominalsymbolrate)

            sequencelength = data_loader.Params.SequenceLengthInterpolated
            num_signal_cur_file = math.floor(sequencelength/self.signal_size)
            train_label_cur_file = data_loader.params(self.label)
            # calculation the total number of training data with varying sequence lengthes in files
            self.num_train_data += num_signal_cur_file
            # extract training label
            self.train_label.append(np.zeros(num_signal_cur_file)+train_label_cur_file)
            # extract training data
            self.train_data.append(data_loader.signal_channel(self.signal_size))
        
        self.train_data = np.array(self.train_data).reshape(-1,4,self.signal_size)
        self.train_label = np.expand_dims(np.array(self.train_label).reshape(-1),axis=-1)

        #test data extraction
    def testsetload(self):
        for filename in self.testfilelist:
            filepath = os.path.join(self.test_path,filename)
            data_loader = extract_OptiSystem(filepath,self.nominalsymbolrate)

            sequencelength = data_loader.Params.SequenceLengthInterpolated
            num_signal_cur_file = math.floor(sequencelength/self.signal_size)
            test_label_cur_file = data_loader.params(self.label)
            # calculation the total number of training data with varying sequence lengthes in files
            self.num_test_data += num_signal_cur_file
            # extract test label
            self.test_label.append(np.zeros(num_signal_cur_file)+test_label_cur_file)
            # extract test data
            self.test_data.append(data_loader.signal_channel(self.signal_size))
        
        self.test_data = np.array(self.test_data).reshape(-1,4,self.signal_size)
        self.test_label = np.expand_dims(np.array(self.test_label).reshape(-1),axis=-1)
    
    def get_batch(self,batch_size):
        index = np.random.randint(0,self.num_train_data,batch_size)
        return self.train_data[index,:],self.train_label[index,:]

    def __call__(self):
        self.trainsetload()
        self.testsetload()

if __name__ == '__main__':
    signal_size = 512
    nominalsymbolrate = 0.5
    label = 'OSNR'
    train_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/train_data/'
    test_path = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/test_data/'
    dataloader = DataLoader(signal_size,nominalsymbolrate,label,train_path,test_path)
    dataloader()
    print('---------train part test---------')
    print('train_data shape:',dataloader.train_data.shape)
    print(dataloader.test_data.shape)
    print('train_label shape:',dataloader.train_label.shape)
    print('---------test part test---------')
    print('test_data shape:',dataloader.test_data.shape)
    print('test_label shape',dataloader.test_label.shape)
    print('---------get_batch() test---------')
    traindata,trainlabel = dataloader.get_batch(2)
    print('traindata in batch:',traindata.shape,'\n','trainlabel in batch:\n',trainlabel)
