from typing import Sequence
from numpy.lib.shape_base import expand_dims
import h5py
import numpy as np
from scipy.interpolate import interp1d
import math
#  Optisystem simulation Dataset extraction
#  This class extractes parameters and signals of four channels simultaneously.
#  signal_channel(signal_size) extractes signal according to the signal_size and   gives the output in the form of (num_signal,4,signal_size) where num_signal=sequencelength/signal_size is the number of the signals can be split in one file according to the signal_size input.
#  signal_pol() extracts signals from 4 channels to 2 polarizations.
#  params() extracts parameter saved in file with its name in string. All the parameters could be extracted at same time with 'All'
class extract_OptiSystem(): 
    def __init__(self,filepath,NominalSymbolRate):

        self.file = h5py.File(filepath,'r')

        self.X_inph = np.array(self.file['InputPort1']['Sampled']['Signal'][:])
        self.X_orth = np.array(self.file['InputPort2']['Sampled']['Signal'][:])
        self.Y_inph = np.array(self.file['InputPort3']['Sampled']['Signal'][:])
        self.Y_orth = np.array(self.file['InputPort4']['Sampled']['Signal'][:])
        
        self.Params = self.file['Params'] #self.Params Group <HDF5 group "/Params" (7 members)>
        
        # do interpolation
        # return two polarization signals in the form of 
        X_in =  self.X_inph[:,0] 
        X_in =+ self.X_orth[:,0]*1j 
        Y_in =  self.Y_inph[:,0] 
        Y_in =+ self.Y_orth[:,0]*1j
    # normalization may be ignored in this paper
        X_in = X_in - np.mean(X_in) 
        Y_in = Y_in - np.mean(Y_in)

        Time_origin = self.file['InputPort1']['Sampled']['Time'][:,0]
        SymbolRate = self.params('SymbolRate')
        SamplingRate_Aim = SymbolRate/NominalSymbolRate
        Tsam_aim = 1/SamplingRate_Aim
        Time_aim = np.arange(0,Time_origin[-1],Tsam_aim)

        InterpolFun = interp1d(Time_origin,X_in,kind='cubic',axis=0)
        self.X_in = InterpolFun(Time_aim)
        InterpolFun = interp1d(Time_origin,Y_in,kind='cubic',axis=0)
        self.Y_in = InterpolFun(Time_aim)
        
        self.Params.SequenceLengthInterpolated = len(self.X_in)

    def signal_channel(self,signal_size):
        # signal_size should be divisible by self.sequencelength
        # return a ndarray of 

        X_in , Y_in = self.signal_pol()
        signal_num = math.floor(self.Params.SequenceLengthInterpolated/signal_size)
        self.Params.SequenceLengthTruncated = signal_num*signal_size
        self.channel1 = np.real(X_in)[0:self.Params.SequenceLengthTruncated]
        self.channel2 = np.imag(X_in)[0:self.Params.SequenceLengthTruncated]
        self.channel3 = np.real(Y_in)[0:self.Params.SequenceLengthTruncated]
        self.channel4 = np.imag(Y_in)[0:self.Params.SequenceLengthTruncated]

        Src = np.array([self.channel1,self.channel2,self.channel3,self.channel4])
        Src = np.squeeze(Src)
        Src = np.transpose(Src).reshape(-1,signal_size,4).transpose(0,2,1)
        return Src
    
    def signal_pol(self):
        return self.X_in,self.Y_in

    def params(self,param_specific):
        # Group self.Params is defined in __init__

        def osnr():
            self.Params.OSNR = self.Params['OSNR'][0,0]
            return self.Params.OSNR
        def rolloff():
            self.Params.Rolloff = self.Params['Rolloff'][0,0]
            return self.Params.Rolloff

        def saperbit():
            self.Params.Saperbit = self.Params['Saperbit'][0,0]
            return self.Params.Saperbit

        def sequencelength():
            self.Params.SequenceLength = int(self.Params['SequenceLength'][0,0])
            return self.Params.SequenceLength

        def signalpower():
            self.Params.SignalPower = self.Params['SignalPower'][0,0]
            return self.Params.SignalPower

        def symbolrate():
            self.Params.SymbolRate = self.Params['SymbolRate'][0,0]
            return self.Params.SymbolRate

        def bitrate():
            self.Params.BitRate = self.Params['BitRate'][0,0]
            return self.Params.BitRate

        # def sequencelengthinterpolated():
        #     SequenceLengthInterpolated = int(self.Params['SequenceLengthInterpolated'][:])
        #     return SequenceLengthInterpolated
        
        ParamsAll = np.array([osnr(),rolloff(),saperbit(),\
                              sequencelength(),signalpower(),\
                              symbolrate(),bitrate()])

        ParamsList = {'OSNR' : osnr(), 'Rolloff' : rolloff(), 'Saperbit' : saperbit(),
                      'SequenceLength' : sequencelength(), 'SignalPower' : signalpower(),
                      'SymbolRate': symbolrate(), 'BitRate' : bitrate(), 'All' : ParamsAll}
                    #   'SequenceLengthInterpolated': sequencelengthinterpolated()}

        
        Params = ParamsList.get(param_specific,'Not Available')

        return Params

if __name__ == '__main__':
    filepath = 'F:/tempo sim data/112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01/112Gbpers_28GBaud_DP-QPSK_20dB_2dBm_131072_400km_No.165.mat'
    NominalSymbolRate = 0.5
    data_loader = extract_OptiSystem(filepath,NominalSymbolRate)
    print('---------__init__() test---------')
    print('__init__() output:')
    print('type of initialization result __init__() (X_inph in example):',type(data_loader.X_inph))
    print('shape of initialization result __init__() (X_inph in example):',data_loader.X_inph.shape)

    # signal_channel() test
    print('---------signal_channel() test---------')
    print('signal_channel() output with nominal symol rate of 0.5:')
    Src = data_loader.signal_channel(512)
    print(type(Src))
    print(Src.shape)
       
    # signal_pol() test
    print('---------signal_pol() test---------')
    print('signal_pol() output with nominal symbol rate of 0.5:')
    x , y = data_loader.signal_pol()
    print(type(data_loader.signal_pol()))
    print('x:\n',x.shape,'\n y:\n',y.shape)
    print('Sequence length interpolated:',data_loader.Params.SequenceLengthInterpolated)

    # params() test
    print('---------param() test---------')
    params = data_loader.params('All')
    print('params() output:\n',params)

    # property test
    print('---------Property test---------')
    print('Params property test (OSNR in example):',data_loader.Params.OSNR)
    print('self.channel (channel1 in example):',data_loader.channel1.shape)

    print('Extraction function test passes!')

