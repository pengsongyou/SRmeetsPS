%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The implementation of the paper:                               %
% "Depth Super-Resolution Meets Uncalibrated Photometric Stereo" %    
% Songyou Peng, Bjoern Haefner, Yvain Queau, Daniel Cremers      %
%                                                                %
% The code can only be used for research purposes.               %
%                                                                %
% CopyRight (C) 2017 Songyou Peng (songyou.peng@in.tum.de)       %
% Computer Vision Group, TUM                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
clc

addpath(strcat(pwd,'/Source/'))
addpath(strcat(pwd,'/Data/'))
%%
sf = 4; % Scaling factor. 2 or 4
data_name = 'mitten';

lambda = 1; % when the result is not desirable, tune between 0.1-100
tol = 5e-3; % Stoping criterion 
max_iter = 10; % Maximum number of iteration

% flag
fill_missing_z = 1; % 1: fill in the missing areas on the input depths   0: otherwise
apply_smooth_filter = 1; % 1: Guided filter to smooth the input depths (!!! use it when there are weird artefacts in z_out, e.g. chapping effects inside, impulsing effect around the boundary)
flag_cmg = 1; % 1: CMG precondition, 0: otherwise (CMG can be downloaded from http://www.cs.cmu.edu/~jkoutis/cmg.html)

%% 
load(strcat(data_name,sprintf('_sf%d',sf)));

[z, rho, N, iter, T_total] = SRPS(I, z0, mask, K, lambda, fill_missing_z, apply_smooth_filter, tol, max_iter, flag_cmg);

