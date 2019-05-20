clc;clear all;close all;
snr=5;
MinPitch= 50;
MaxPitch= 500;
pitch_step=2;
N=240;
fs=8000;

Number_source=2;
f0=[160+(rand-1/2)*pitch_step;240+(rand-1/2)*pitch_step];
L=[3+floor(7*rand);3+floor(7*rand)];

Z1 = CreateDictionary(N, L(1), f0(1)/fs);
a1 = exp(1i*2*pi*rand(L(1),1));

Z2 = CreateDictionary(N, L(2), f0(2)/fs);
a2 = exp(1i*2*pi*rand(L(2),1));

x=Z1*a1+Z2*a2;
y =x+addnoise_strict_snr(x,randn(size(x))+1i*randn(size(x)),snr); %creates signal
% y=y/max(abs(y));
[f0_estimate]=PE_BSBL_Cluster(y,fs,MinPitch,MaxPitch,pitch_step,Number_source);
ture_f0=f0(:)'
esitimated_f0=f0_estimate(:)'

%%%%%%%% Utility functions
 function output_noise=addnoise_strict_snr(sig,input_noise,snr)
noise=input_noise;
noise_std_var=sqrt(10^(-snr/10)*(sig'*sig)/(noise'*noise));
output_noise=noise_std_var*noise;
 end
function [W]=CreateDictionary(DataLength, MaxHarms,PitchGrid)%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
NoPitches=length(PitchGrid);
W = [];
for i = 1:NoPitches
    for j=1:MaxHarms
    W = [W exp(1i*PitchGrid(i)*j*2*pi*[0:DataLength-1]')];%/sqrt(DataLength)];
    end
end
end
