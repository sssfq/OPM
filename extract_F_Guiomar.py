import h5py
#  Fernando Guiomar Dataset extraction
#  Zendo url:https://zenodo.org/record/1094985#.YMHAVr7itPZ
class extract_F_Guiomar: 
    def __init__(self):

        self.file = h5py.File(filepath,'r')


    def signal(filepath):

        sig = self.file['Y'][:]

        return sig

    def params(filepath):

        params = self.file['AcqParams']

        return params

if __name__ == '__main__':
    filepath = 'D:/OneDrive/Postgraduate/Optical performance monitoring (OPM)/proj/Rx_Nrec10_run1.mat'
    
    sig = extract_F_Guiomar.signal(filepath)
    print('Signal:\n',sig)
    print('Signal size is:\n',sig.shape)
    params = extract_F_Guiomar.params(filepath)
    print('Parameters:',params.keys())
    
    print('Extraction function test passes!')
