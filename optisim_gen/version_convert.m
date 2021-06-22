clear all

Rolloff = [0.01 0.03 0.05 0.08 0.1 0.2 0.3 0.4 0.5 0.6];
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
for i = 1:length(Rolloff)
    Path = 'F:\tempo sim data\224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6\';
    File = dir(fullfile(Path,strcat('224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_',num2str(Rolloff(i)),'*')));
    filenames = {File.name};  %get corresponding 400 files of the rolloff in current loop
    for ii = 1:length(filenames)
        load(strcat(Path,filenames{ii}))
        
        Params.Rolloff = rolloff;
        Params.Saperbit = Samplesperbit;
        Params.SequenceLength = SequenceLength;
        Params.SignalPower = SignalPower;
        Params.SymbolRate = SymbolRate;
        Params.OSNR = OSNR;
        Params.BitRate = SymbolRate*8;
        
        save(filenames{ii},'InputPort1','InputPort2','InputPort3','InputPort4', ...
                           'Params','-v7.3')
    end
end
