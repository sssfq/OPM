close all;
format short;

Params.Rolloff = rolloff;
Params.Saperbit = Samplesperbit;
Params.SequenceLength = SequenceLength;
Params.SignalPower = SignalPower;
Params.SymbolRate = SymbolRate;
Params.OSNR = OSNR;
Params.BitRate = SymbolRate*8;
Params.ChromaticDispersion = ChromaticDispersion;
Params.NumberOfLoops = NumberOfLoops;


filename = sprintf(strcat("F:\\tempo sim data\\112Gbpers_28GBaud_DP-QPSK_1Saperb_2dBm_0.01\\", ...
                           num2str(BitRate/1e9),"Gbpers_", ...
                           num2str(SymbolRate/1e9),"GBaud_DP-16QAM_", ...
                           num2str(Samplesperbit),"Saperb_", ...
                           num2str(OSNR),"dB_", ...
                           num2str(SignalPower),"dBm_", ...
                           num2str(SequenceLength), ...
                           num2str(80*NumberOfLoops),"km_", ...
                           "_%s.mat"),datestr(now,"mmmmddyyyyHHMMSSFFF") ...
                    );
save(filename,'InputPort1','InputPort2','InputPort3','InputPort4','Params','-v7.3')