clear all

OSNR = 10:2:20;
% %% Convert current .mat to version 7.3
% for i = 1:length(Rolloff)
%     Path = 'F:\tempo sim data\224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_0.01-0.6\';
%     File = dir(fullfile(Path,strcat('224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_',num2str(Rolloff(i)),'*')));
%     filenames = {File.name}  %get corresponding 400 files of the rolloff in current loop
% 
%     for ii = 1:length(filenames)
%         load(strcat(Path,filenames{ii}))
%         file_name =strcat('224Gbpers_28GBaud_DP-16QAM_1Saperb_18dB_2dBm_',num2str(Rolloff(i)),'_','No.',num2str(ii),'.mat')
%         save(file_name,'InputPort1','InputPort2','InputPort3','InputPort4', ...
%                       'OSNR','rolloff','Samplesperbit','SequenceLength','SymbolRate','SignalPower', ...
%                       '-v7.3')
%     end
% end

%% Structure
for i = 1:length(OSNR)
    Path = 'F:\tempo sim data\112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01\';
    File = dir(fullfile(Path,strcat('112Gbpers_28GBaud_DP-16QAM_1Saperb_',num2str(OSNR(i)),'dB_2dBm_*')));
    filenames = {File.name};  %get corresponding 200 files of the OSNR in current loop
    for ii = 1:length(filenames)
        load(strcat(Path,filenames{ii}))
        
%         Params.Rolloff = rolloff;
%         Params.Saperbit = Samplesperbit;
%         Params.SequenceLength = SequenceLength;
%         Params.SignalPower = SignalPower;
%         Params.SymbolRate = SymbolRate;
%         Params.OSNR = OSNR;
%         Params.BitRate = SymbolRate*4;
        
%         downsampling
%         InXo = InputPort1.Sampled.Signal + 1i * InputPort2.Sampled.Signal+InputPort1.Noise.Signal+1i*InputPort2.Noise.Signal;
%         InYo = InputPort3.Sampled.Signal + 1i * InputPort4.Sampled.Signal+InputPort3.Noise.Signal+1i*InputPort4.Noise.Signal;
%         InXo = InXo-mean(InXo);
%         InYo = InYo-mean(InYo);
%         Time = InputPort1.Sampled.Time;
% 
%         SymbolRate = 28e9;
%         SampleRate = 112e9;
%         SampleRate_Aim = 56e9;
%         Tsam = 1/SampleRate_Aim;
%         Time_Aim = 0:Tsam:Time(end);
% 
%         InXo = interp1(Time,InXo,Time_Aim, 'spline');
%         InYo = interp1(Time,InYo,Time_Aim, 'spline');
        
        filenames_no = strcat('112Gbpers_28GBaud_DP-QPSK_',num2str(OSNR(i)),'dB_2dBm_',num2str(Params.SequenceLength),'_400km_No.',num2str(ii),'.mat')
        save(filenames_no,'InputPort1','InputPort2','InputPort3','InputPort4', ...
                           'Params','-v7.3')
        delete(filenames{ii})
    end
end
