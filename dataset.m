% ShortMessageLength is defalut 5
N = 15;
K = 11;
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

%.csv可以更改为.txt等
filename='resylt.csv';
fid=fopen(filename,'w');
count=0;
for index=1:1000
    
    i = randi([0 2],1,1);
    if i == 0
        data = randi([0 1],1005,1);
        fprintf("%d", data);
        fprintf(" ndims=%d\n", ndims(data));
        encoded = bch_encoder(data);
    elseif i == 1
        data = randi([0 1],1005,1);
        fprintf("%d", data);
        fprintf(" ndims=%d\n", ndims(data));
        encoded = rs_encoder(data);
    else
        data = randi([0 1],1000,1);
        fprintf("%d", data);
        fprintf(" ndims=%d\n", ndims(data));
        encoded = convolutional_encoder(data);
    end
    
    count=count+1;
        
    fprintf("%d", encoded);
    fprintf(" %d", i);
    fprintf("\n");
    %   ','是分隔符
    fprintf(fid, '%d, %s, %d\n',count, num2str(encoded), i);
end