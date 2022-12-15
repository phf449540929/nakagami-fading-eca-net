clear;close all;clc

numSymb = 1e4;
EbNo = 0:.5:0;
M = 2; %codeRate = 1/2;
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);

for i=1:length(EbNo)
    filename=sprintf('%s%.1f%s', './dataset/length/conv_4/dataset-length-conv-', EbNo(i), 'db.csv');
    disp(filename)
    fid=fopen(filename,'w');
    fprintf(fid, ',text,label\n');
    
    EbNoDemo =EbNo(i); 
    EsN0 = EbNoDemo + 10*log10(k);
    
    for index=1:12000
        j = randi([0 5],1,1);
        if j == 0
            codeRate=1/2;
            codeRateStr='R1/2';
            k=2;
            label = 'length=2';
        elseif j == 1
            codeRate=2/3;
            codeRateStr='R2/3';
            k=4;
            label = 'length=3';
        elseif j == 2
            codeRate=3/4;
            codeRateStr='R3/4';
            k=6;
            label = 'length=4';
        elseif j == 3
            codeRate=5/6;
            codeRateStr='R5/6';
            k=10;
            label = 'length=6';
         elseif j == 4
             codeRate=6/7;
             codeRateStr='R6/7';
             k=12;
             label = 'length=7';
         else
             codeRate=7/8;
             codeRateStr='R7/8';
             k=14;
             label = 'length=8';
        end

        msg_orig = randi([0 1], numSymb*2, 1);
        msg_enc = convenc(msg_orig, trellis); % 编码
        % 将标准编码后的序列进行打孔发送，提高码率
        msg_enc_punc = conv_tx_puncture(msg_enc,codeRateStr);  % 打孔后的序列即为要发送的序列
        msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
        msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪
        % 取复基带信号的实部进行反打孔，将删除的位置补零，仿真中这就是接收到的序列
        demodulated = pskdemod(msg_rx, M);
        
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
