clear;close all;clc
numSymb = 1e3;
EbNo = 0:.5:8;
codeRate=1/2;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb, 1);
    msg_enc = convenc(msg_orig, trellis);
    msg_tx = pskmod(msg_enc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪
    [x,qcode] = quantiz(real(msg_rx),[ -.5  0  .5 ],3:-1:0); % 软判决译码映射 qcode的值在0到2^2-1之间
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2);
    [err,ber1(i)] = biterr(msg_dec(tblen+1:end),msg_orig(1:end-tblen)); %误比特率
    fprintf("%d\n", length(msg_enc));
end

%%%%%%%%%%%%%%%%码率为2/3的代码段%%%%%%%%%%%%%%%%%%%
EbNo = 0:.5:8;
codeRate=2/3;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb*4, 1);
    msg_enc = convenc(msg_orig, trellis); % 编码
    % 将标准编码后的序列进行打孔发送，提高码率
    msg_enc_punc = conv_tx_puncture(msg_enc,'R2/3');  % 打孔后的序列即为要发送的序列
    fprintf("%d\n", length(msg_enc_punc));
    msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪
    % 取复基带信号的实部进行反打孔，将删除的位置补零，仿真中这就是接收到的序列
    qcode_depunc = conv_rx_depuncture(real(msg_rx),'R2/3');     
    [x,qcode] = quantiz(qcode_depunc,[ -.5  0  .5 ],3:-1:0); % 软判决译码映射  qcode的值在0到2^2-1之间
    
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2); % Viterbi 软判决译码
    mlen = min(length(msg_dec),length(msg_orig))-tblen;
    [acor,lag] = xcorr(msg_dec,msg_orig); % 求相关计算起点，进而计算误比特率
    [valm,I]=max(abs(acor));
    delay=lag(I);
    [err,ber2(i)] = biterr(msg_dec(delay+1:delay+mlen),msg_orig(1:mlen)); %误比特率
end

%%%%%%%%%%%%%%%%码率为3/4的代码段%%%%%%%%%%%%%%%%%%%
EbNo = 0:.5:8;
codeRate=3/4;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb*6, 1);
    msg_enc = convenc(msg_orig, trellis); % 编码
    msg_enc_punc = conv_tx_puncture(msg_enc,'R3/4'); 
    fprintf("%d\n", length(msg_enc_punc));
    msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪
    
    qcode_depunc = conv_rx_depuncture(real(msg_rx),'R3/4');
    [x,qcode] = quantiz(qcode_depunc,[ -.5  0  .5 ],3:-1:0);
    
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2);
    mlen = min(length(msg_dec),length(msg_orig))-tblen;
    [acor,lag] = xcorr(msg_dec,msg_orig);
    [valm,I]=max(abs(acor));
    delay=lag(I);
    [err,ber3(i)] = biterr(msg_dec(delay+1:delay+mlen),msg_orig(1:mlen)); %误比特率
end

%%%%%%%%%%%%%%%%码率为5/6的代码段%%%%%%%%%%%%%%%%%%%
EbNo = 0:.5:8;
codeRate=5/6;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb*10, 1);
    msg_enc = convenc(msg_orig, trellis); % 编码
    msg_enc_punc = conv_tx_puncture(msg_enc,'R5/6'); 
    fprintf("%d\n", length(msg_enc_punc));
    msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪    
    
    qcode_depunc = conv_rx_depuncture(real(msg_rx),'R5/6');
    [x,qcode] = quantiz(qcode_depunc,[ -.5  0  .5 ],3:-1:0);  
    
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2);
    mlen = min(length(msg_dec),length(msg_orig))-tblen;
    [acor,lag] = xcorr(msg_dec,msg_orig);
    [valm,I]=max(abs(acor));
    delay=lag(I);
    [err,ber4(i)] = biterr(msg_dec(delay+1:delay+mlen),msg_orig(1:mlen)); %误比特率
end

%%%%%%%%%%%%%%%%码率为6/7的代码段%%%%%%%%%%%%%%%%%%%
EbNo = 0:.5:8;
codeRate=6/7;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb*12, 1);
    msg_enc = convenc(msg_orig, trellis); % 编码
    msg_enc_punc = conv_tx_puncture(msg_enc,'R6/7'); 
    fprintf("%d\n", length(msg_enc_punc));
    msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪 
    
    qcode_depunc = conv_rx_depuncture(real(msg_rx),'R6/7');
    [x,qcode] = quantiz(qcode_depunc,[ -.5  0  .5 ],3:-1:0);  
    
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2);
    mlen = min(length(msg_dec),length(msg_orig))-tblen;
    [acor,lag] = xcorr(msg_dec,msg_orig);
    [valm,I]=max(abs(acor));
    delay=lag(I);
    [err,ber5(i)] = biterr(msg_dec(delay+1:delay+mlen),msg_orig(1:mlen)); %误比特率
end


%%%%%%%%%%%%%%%%码率为7/8的代码段%%%%%%%%%%%%%%%%%%%
EbNo = 0:.5:8;
codeRate=7/8;
M = 2; %codeRate = 1/2; 
constlen = 7; k = log2(M); codegen = [171 133];
tblen = 32;     % traceback length
trellis = poly2trellis(constlen, codegen);
for i=1:length(EbNo) 
    EbNoDemo =EbNo(i); EsN0 = EbNoDemo + 10*log10(k);
    msg_orig = randi([0 1], numSymb*14, 1);
    msg_enc = convenc(msg_orig, trellis); % 编码
    msg_enc_punc = conv_tx_puncture(msg_enc,'R7/8'); 
    fprintf("%d\n", length(msg_enc_punc));
    msg_tx = pskmod(msg_enc_punc, M); % BPSK调制 - 复基带信号
    msg_rx = awgn(msg_tx, EsN0-10*log10(1/codeRate),'measured'); % 加噪  
    
    qcode_depunc = conv_rx_depuncture(real(msg_rx),'R7/8');
    [x,qcode] = quantiz(qcode_depunc,[ -.5  0  .5 ],3:-1:0);   
    
    msg_dec = vitdec(qcode.',trellis,tblen,'cont','soft',2);
    mlen = min(length(msg_dec),length(msg_orig))-tblen;
    [acor,lag] = xcorr(msg_dec,msg_orig);
    [valm,I]=max(abs(acor));
    delay=lag(I);
    [err,ber6(i)] = biterr(msg_dec(delay+1:delay+mlen),msg_orig(1:mlen)); %误比特率
end

semilogy(EbNo,ber1, 'b*-',EbNo,ber2, 'r-o',EbNo,ber3, 'k-s',EbNo,ber4, 'd-m', EbNo,ber5, 'd-m', EbNo,ber6, 'd-m');%codeRate=5/6
legend('codeRate=1/2', 'codeRate=2/3','codeRate=3/4','codeRate=5/6','codeRate=6/7','codeRate=7/8');
xlabel('Eb/N0');ylabel('误码率');
