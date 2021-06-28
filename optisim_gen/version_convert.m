clear all

OSNR = 10:2:20;
%% Convert current .mat to version 7.3
for i = 1:length(Rolloff)
    Path = 'F:\tempo sim data\224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6\';
    File = dir(fullfile(Path,strcat('224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_',num2str(Rolloff(i)),'*')));
    filenames = {File.name}  %get corresponding 400 files of the rolloff in current loop

    for ii = 1:length(filenames)
        load(strcat(Path,filenames{ii}))
        file_name =strcat('224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_',num2str(Rolloff(i)),'_','No.',num2str(ii),'.mat')
        save(file_name,'InputPort1','InputPort2','InputPort3','InputPort4', ...
                      'OSNR','rolloff','Samplesperbit','SequenceLength','SymbolRate','SignalPower', ...
                      '-v7.3')
    end
end

%% Structure
for i = 1:length(OSNR)
    Path = 'F:\tempo sim data\112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01\';
    File = dir(fullfile(Path,strcat('112Gbpers_28GBaud_DP-16QAM_1Saperb_',num2str(OSNR),'dB_2dBm_*')));
    filenames = {File.name};  %get corresponding 200 files of the OSNR in current loop
    for ii = 1:length(filenames)
        load(strcat(Path,filenames{ii}))
        
        Params.Rolloff = rolloff;
        Params.Saperbit = Samplesperbit;
        Params.SequenceLength = SequenceLength;
        Params.SignalPower = SignalPower;
        Params.SymbolRate = SymbolRate;
        Params.OSNR = OSNR;
        Params.BitRate = SymbolRate*8;
        filename_no = strcat('112Gbpers_28GBaud_DP-16QAM_1Saperb_',num2str(OSNR),'dB_2dBm_131072_400km_No.',num2str(ii),'.mat')
        save(filenames{ii},'InputPort1','InputPort2','InputPort3','InputPort4', ...
                           'Params','-v7.3')
    end
end
