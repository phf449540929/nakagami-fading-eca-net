%.csv可以更改为.txt等
slCharacterEncoding('UTF-8')

for i = -10:2:20
    snr = i;
    
    coding = 'ldpc';
    filename=sprintf('%s%s%s%d%s', 'dataset-awgn-', coding, '-', snr, 'db.csv');
    disp(filename)
    fid=fopen(filename,'w');
    fprintf(fid, ',text,label\n');

    noise_var = 10.^(-snr/10);
    % modulator = comm.QPSKModulator('BitInput',true);
    modulator = comm.BPSKModulator();
    channel = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)', 'SNR', snr);
    % demodulator_1= comm.QPSKDemodulator('BitOutput',true);
    % demodulator= comm.QPSKDemodulator('BitOutput',true, 'DecisionMethod','Log-likelihood ratio', 'VarianceSource', 'Input port');
    demodulator= comm.BPSKDemodulator(...
        'DecisionMethod','Log-likelihood ratio',...
        'VarianceSource', 'Input port');

    for index=1:1000
        if strcmp(coding, 'conv')
            [encoded, label] = encode_conv_2();
        elseif strcmp(coding, 'ldpc')
            [encoded, label] = encode_ldpc();
        elseif strcmp(coding, 'polar')
            [encoded, label] = encode_polar();
        elseif strcmp(coding, 'turbo')
            [encoded, label] = encode_turbo();
        end
    
        modulator.release();
        demodulator.release();
%       disp(size(encoded));
        modulated = step(modulator, encoded);
        signal = step(channel, modulated);
%       if index == 1
%           scatterplot(signal);
%       end
%       demodulated_1 = step(demodulator_1, signal);
        demodulated = step(demodulator, signal, noise_var);
    
%       fprintf("%f", demodulated);
        fprintf(" %s %d\n", label, length(demodulated(:)));
        %   ','是分隔符
        fprintf(fid, '%d, ',index - 1);
        for j = demodulated
            fprintf(fid, "%f/", j);
        end
        fprintf(fid, ', %s\n', label);
    
    end 
end


    