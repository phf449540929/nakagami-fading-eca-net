clc
clear
close all 

% Igor Martins Rocha
% Simulation of nakagami-m fading channel

m = 1;      % Controls the format, the higher the m, the weaker the fading
w = 1;      % Controls the spreading
% snr = 20;

% --------- Plotar FDP --------------------------- %
% x = 0:0.01:5;
% f = (2*(m^m)*x.^(2*m-1).*exp((-m/w)*x.^2))/(gamma(m)*(w^m)); % FDP
% plot (x,f)
% axis([0 3 0 1.5])
% title('Probability Density Function')

% --------- Nakagami-m distribution --------------%
% fs = 100;                                   %Signal frequency
% fa = 20*fs;                                 %Sampling frequency
% t = 0:1/fa:100;
% y1 = sin (2*pi*fs*t);     

% y2 = awgn (y1, snr, 'measured');            % Signal with noise

% pd = makedist('Nakagami','mu',m,'omega',w); 
% r = random(pd,length(t),1)';                

% y3  = r.*y1;                                % Signal with fading
% y4 = awgn (y3, snr,'measured');             % Signal with noise and fading

% ---------- Constellation --------------------%

for i = -10:2:20
    
    snr = i;
    filename=sprintf('%s%d%s', './dataset/awgn/dataset-awgn-ldpc-', snr, 'db.csv');
    disp(filename)
    fid=fopen(filename,'w');
    fprintf(fid, ',text,label\n');
    
    noise_var = 10.^(-snr/10);
    modulator = comm.BPSKModulator();
    demodulator= comm.BPSKDemodulator(...
            'DecisionMethod','Log-likelihood ratio',...
            'VarianceSource', 'Input port');

%     M = 4;
%     sym = 1000; % Number of symbols
%     
%     data = randi([0 M-1],sym,1);  


    for index=1:1000

        [encoded, label] = ldpc_ieee();

        modulator.release();
        demodulator.release();

        txSig = step(modulator, encoded);
        % txSig = pskmod(data,M,pi/M);  

%         r = random (pd, length(encoded), 1);  
        % r = random (pd, sym, 1);      

%         rxd = r.*txSig;                       % Adding the fading
        % rxn = awgn (txSig, snr, 'measured');   % Adding the noise
        rxdn = awgn(txSig, snr, 'measured');     % Noise and fading

        demodulated = step(demodulator, rxdn, noise_var);

        fprintf(" %s %d\n", label, length(demodulated(:)));
        fprintf(fid, '%d, ',index - 1);
        for j = demodulated
            fprintf(fid, "%f/", j);
        end
        fprintf(fid, ', %s\n', label);
    end
end

% scatterplot(txSig)
% title('Constellation')
% 
% scatterplot(rxn)
% title('Constellation with noise')
% 
% scatterplot(rxd)
% title('Constellation with fading')
% 
% scatterplot(rxdn)
% title('Constellation with noise and fading')


% --------- Plotting the graphics ---------- %
% figure();
% subplot (2,2,1)
% plot (t,y1);
% title('Signal')
% axis([0 5/fs -2 2])
% 
% subplot (2,2,2)
% plot (t,y2);
% title('Signal with noise')
% axis([0 5/fs -2 2])
% 
% subplot (2,2,3)
% plot (t,y3);
% title('Signal with fading')
% axis([0 5/fs -2 2])
% 
% subplot(2,2,4)
% plot (t,y4);
% title('Signal with fading and noise')
% axis([0 5/fs -2 2])