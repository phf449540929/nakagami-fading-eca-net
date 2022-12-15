clear;close all;clc

numSymb = 1e4;
EbNo = 0:.5:12;
M = 2; %codeRate = 1/2;
k = log2(M);

% ShortMessageLength is defalut 5
N = 15;
K = 7;
gp = bchgenpoly(N,K);
S = 5;
bch_encoder = comm.BCHEncoder(N,K,gp, S);

% RS codeword length
% RS message length
N = 7;
K = 5;
rs_encoder = comm.RSEncoder('BitInput',true,'CodewordLength',N,'MessageLength',K);

constlen = 7;
codegen = [171, 133];
% G1=171 , G2=133.
trellis = poly2trellis(constlen, codegen);
convolutional_encoder = comm.ConvolutionalEncoder('TrellisStructure',trellis);

load H1944_12.mat;
load G1944_12.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_12 = G;
N_1944_12 = N;

L = 5000;
intrlvrIndices = randperm(L);
trellis = poly2trellis(5,[33 23 17],33);
turbo_encoder = comm.TurboEncoder(trellis, intrlvrIndices);

K = 144;
E_150 = 150;

for i=1:length(EbNo)
    
    filename=sprintf('%s%.1f%s', './dataset/type/dataset-type-', EbNo(i), 'db.csv');
    disp(filename)
    fid=fopen(filename,'w');
    fprintf(fid, ',text,label\n');
    
    EbNoDemo =EbNo(i); 
    EsN0 = EbNoDemo + 10*log10(k);
    
    noise_var = 10.^(-i/10);
    modulator = comm.BPSKModulator();
    channel = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)', 'SNR', i);
    demodulator= comm.BPSKDemodulator();
    
    for index=1:12000
        
        a = randi([0 5],1,1);
        if a == 0
            data = randi([0 1],10050,1);
            encoded = bch_encoder(data);
            label = 'bch';
        elseif a == 1
            data = randi([0 1],20100,1);
            encoded = rs_encoder(data);
            label = 'rs';
        elseif a == 2
            data = randi([0 1],10000,1);
            encoded = convolutional_encoder(data);
            label = 'conv';
        elseif a == 3
            for j = 1:1:10
                s=randi([0,1],1,N_1944_12);
                s_l=mod(s*G_1944_12,2);
                s_2 = [s_l]';
                if j == 1
                    encoded = s_2;
                else
                    encoded = [encoded; s_2];
                end
            end
            label = 'ldpc';
        elseif a == 4
            data = randi([0 1],L,1);
            encoded = turbo_encoder(data);
            label = 'turbo';
        else
            for j=1:100
                msg = randi([0 1],K,1,'int8');
                enc = nrPolarEncode(msg,E_150);
                if j == 1
                    encoded = enc;
                else
                    encoded = [encoded; enc];
                end
            end
            label = 'polar';
        end
        
        % disp(encoded)
        
        % msg_tx = pskmod(encoded, M); % BPSK调制 - 复基带信号
        % msg_rx = awgn(msg_tx, EsN0-10*log10(1/0.5),'measured'); % 加噪
        % 取复基带信号的实部进行反打孔，将删除的位置补零，仿真中这就是接收到的序列
        % demodulated = pskdemod(msg_rx, M);
        
        modulator.release();
        demodulator.release();
        modulated = step(modulator, encoded);
        signal = step(channel, modulated);
        demodulated = step(demodulator, signal);
        
        %       fprintf("%f", demodulated);
        fprintf("%d %s %d ",index - 1, label, length(demodulated));
        demodulated = demodulated(1:16384);
        fprintf("%d\n", length(demodulated));
        %   ','是分隔符
        fprintf(fid, '%d, ',index - 1);
        for j = demodulated
            fprintf(fid, "%d", j);
        end
        fprintf(fid, ', %s\n', label);
    end
end
