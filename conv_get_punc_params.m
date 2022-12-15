function [punc_patt, punc_patt_size] = conv_get_punc_params(code_rate)
    % 打孔参数
    if strcmp(code_rate,'R7/8') % 10个删除4个
       punc_patt=[1 2 3 5 7 10 11 14];
       punc_patt_size = 14;
    elseif strcmp(code_rate,'R6/7') % 10个删除4个
       punc_patt=[1 2 3 5 8 9 12];
       punc_patt_size = 12;
    elseif strcmp(code_rate,'R5/6') % 10个删除4个
       punc_patt=[1 2 3 6 7 10];
       punc_patt_size = 10;
    elseif strcmp(code_rate,'R4/5') % 8个删除3个，由1/2得到4/5
       punc_patt=[1 2 3 5 7];
       punc_patt_size = 8;
    elseif strcmp(code_rate,'R3/4')         % 六个删除两个，由1/2得到3/4
       % R=3/4, Puncture pattern: [1 2 3 x x 6], x = punctured 
       punc_patt=[1 2 3 6];
       punc_patt_size = 6;
    elseif strcmp(code_rate, 'R2/3')    % 四个删除一个，由1/2得到2/3
       % R=2/3, Puncture pattern: [1 2 3 x], x = punctured 
       punc_patt=[1 2 3]; 
       punc_patt_size = 4;
    elseif strcmp(code_rate, 'R1/2')    % 标准编码器码流，不必打孔
       % R=1/2, Puncture pattern: [1 2 3 4 5 6], x = punctured 
       punc_patt=[1 2 3 4 5 6];
       punc_patt_size = 6;
    else
       error('Undefined convolutional code rate');
    end
