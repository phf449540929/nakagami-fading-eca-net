function [encoded,label] = ldpc_ieee_length()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

load H640_25.mat;
load G640_25.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_640_25 = G;
N_640_25 = N;

load H648_12.mat;
load G648_12.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_648_12 = G;
N_648_12 = N;

load H1296_12.mat;
load G1296_12.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1296_12 = G;
N_1296_12 = N;

load H1944_34.mat;
load G1944_34.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_34 = G;
N_1944_34 = N;

load H2560_25.mat;
load G2560_25.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_2560_25 = G;
N_2560_25 = N;

load H4512_12.mat;
load G4512_12.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_4512_12 = G;
N_4512_12 = N;


i = randi([0 3],1,1);
for j = 1:1:30
    if i == 0
        s=randi([0,1],1,N_648_12);
        s_l=ldpc_encode(s,G_648_12);
        s_2 = [s_l]';
        label = 'length=648';
    elseif i == 1
        s=randi([0,1],1,N_1296_12);
        s_l=ldpc_encode(s,G_1296_12);
        s_2 = [s_l]';
        label = 'length=1296';
    elseif i == 2
        s=randi([0,1],1,N_1944_34);
        s_l=ldpc_encode(s,G_1944_34);
        s_2 = [s_l]';
        label = 'length=1944';
    else
        s=randi([0,1],1,N_2560_25);
        s_l=ldpc_encode(s,N_2560_25);
        s_2 = [s_l]';
        label = 'length=2304';
%     elseif i == 4
%         s=randi([0,1],1,N_2560_25);
%         s_l=ldpc_encode(s,G_2560_25);
%         s_2 = [s_l]';
%         label = 'length=1944';
%     else
%         s=randi([0,1],1,N_4512_12);
%         s_l=ldpc_encode(s,G_4512_12);
%         s_2 = [s_l]';
%         label = 'length=2304';
    end
    
    if j == 1
        encoded = s_2;
    else
        encoded = [encoded; s_2];
    end
    
    if length(encoded(:)) >= 16384
        break;
    
end



% disp(label)
% disp(length(encoded(:)))
% encoded = encoded(1:2040);

end

