import h5py
import numpy as np
#  Optisystem simulation Dataset extraction
#  This class extractes parameters and signals of four channels simultaneously.
#  signal_channel(signal_size) extractes signal according to the signal_size and   gives the output in the form of (num_signal,4,signal_size) where num_signal=sequencelength/signal_size is the number of the signals can be split in one file according to the signal_size input.
#  signal_pol() extracts signals from 4 channels to 2 polarizations.
#  params() extracts parameter saved in file with its name in string. All the parameters could be extracted at same time with 'All'
class extract_OptiSystem(): 
    def __init__(self,filepath):

        self.file = h5py.File(filepath,'r')

        self.X_inph = self.file['InputPort1']['Sampled']['Signal'][:]
        self.X_orth = self.file['InputPort2']['Sampled']['Signal'][:]
        self.Y_inph = self.file['InputPort3']['Sampled']['Signal'][:]
        self.Y_orth = self.file['InputPort4']['Sampled']['Signal'][:]

        self.X_inph_Noise = self.file['InputPort1']['Noise']['Signal'][:]
        self.X_orth_Noise = self.file['InputPort2']['Noise']['Signal'][:]
        self.Y_inph_Noise = self.file['InputPort3']['Noise']['Signal'][:]
        self.Y_orth_Noise = self.file['InputPort4']['Noise']['Signal'][:]

        self.channel1 = self.X_inph #+ self.X_inph_Noise
        self.channel2 = self.X_orth #+ self.X_orth_Noise
        self.channel3 = self.Y_inph #+ self.Y_inph_Noise
        self.channel4 = self.Y_orth #+ self.Y_orth_Noise

        self.Params = self.file['Params'] #self.Params Group <HDF5 group "/Params" (7 members)>

    def signal_channel(self,signal_size):
        # signal_size should be divisible by self.sequencelength

        Src = np.array([self.channel1,self.channel2,self.channel3,self.channel4])
        Src = np.squeeze(Src)
        Src = np.transpose(Src).reshape(-1,signal_size,4).transpose(0,2,1)
        return Src
    
    def signal_pol(self):

        X_in =  self.X_inph 
        X_in =+ self.X_orth*1j 
        Y_in =  self.Y_inph 
        Y_in =+ self.Y_orth*1j
        
        return X_in,Y_in

    def params(self,param_specific):
        # Group self.Params is defined in __init__

        def osnr():
            OSNR = self.Params['OSNR'][:]
            return OSNR[0,0]
        
        def rolloff():
            Rolloff = self.Params['Rolloff'][:]
            return Rolloff[0,0]

        def saperbit():
            Saperbit = self.Params['Saperbit'][:]
            return Saperbit[0,0]

        def sequencelength():
            SequenceLength = int(self.Params['SequenceLength'][:])
            return SequenceLength

        def signalpower():
            SignalPower = self.Params['SignalPower'][:]
            return SignalPower[0,0]

        def symbolrate():
            SymbolRate = self.Params['SymbolRate'][:]
            return SymbolRate[0,0]

        def bitrate():
            BitRate = self.Params['BitRate'][:]
            return BitRate[0,0]
        
        ParamsAll = np.array([osnr(),rolloff(),saperbit(),\
                              sequencelength(),signalpower(),\
                              symbolrate(),bitrate()])

        ParamsList = {'OSNR' : osnr(), 'Rolloff' : rolloff(), 'Saperbit' : saperbit(),
                      'SequenceLength' : sequencelength(), 'SignalPower' : signalpower(),
                      'SymbolRate': symbolrate(), 'BitRate' : bitrate(), 'All' : ParamsAll
                      }

        
        Params = ParamsList.get(param_specific,'Not Available')

        return Params

if __name__ == '__main__':
    filepath = 'F:/tempo sim data/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6/224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01_No.100.mat'
    
    data_loader = extract_OptiSystem(filepath)

    print(data_loader.X_inph)

    # signal_channel() test
    print('signal_channel() output:')
    print(data_loader.signal_channel(512).shape)
       
    # signal_pol() test
    print('signal_pol() output:')
    x , y = data_loader.signal_pol()
    print('x:\n',x,'\n y:\n',y)

    # params() test
    params = data_loader.params('All')
    print('params() output:\n',params)

    print('Extraction function test passes!')

