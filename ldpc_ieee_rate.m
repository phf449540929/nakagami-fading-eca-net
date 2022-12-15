function [encoded,label] = ldpc_ieee_rate()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

load H1944_12.mat;
load G1944_12.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_12 = G;
N_1944_12 = N;

load H1944_23.mat;
load G1944_23.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_23 = G;
N_1944_23 = N;

load H1944_34.mat;
load G1944_34.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_34 = G;
N_1944_34 = N;

load H1944_56.mat;
load G1944_56.mat;
[m,n]=size(H);
R=(n-m)/n;
N=size(G,1);
G_1944_56 = G;
N_1944_56 = N;


i = randi([0 3],1,1);
for j = 1:1:10
    if i == 0
        s=randi([0,1],1,N_1944_12);
        s_l=ldpc_encode(s,G_1944_12);
        s_2 = [s_l]';
        label = 'rate=12';
    elseif i == 1
        s=randi([0,1],1,N_1944_23);
        s_l=ldpc_encode(s,G_1944_23);
        s_2 = [s_l]';
        label = 'rate=23';
    elseif i == 2
        s=randi([0,1],1,N_1944_34);
        s_l=ldpc_encode(s,G_1944_34);
        s_2 = [s_l]';
        label = 'rate=34';
    else
        s=randi([0,1],1,N_1944_56);
        s_l=ldpc_encode(s,G_1944_56);
        s_2 = [s_l]';
        label = 'rate=56';
    end
    
    if j == 1
        encoded = s_2;
    else
        encoded = [encoded; s_2];
    end

end



% disp(label)
% disp(length(encoded(:)))
% encoded = encoded(1:2040);

end

