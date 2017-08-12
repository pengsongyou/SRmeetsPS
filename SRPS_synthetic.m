function [z_out, e_rmse,e_mae, iter, T_total] = SRPS_synthetic(I, z0, z, N_ground, mask, K, lambda, fill_missing_z, apply_smooth_filter, tol, max_iter, flag_cmg, do_display)
% Implementation of the paper "Depth Super-Resolution Meets Uncalibrated
% Photometric Stereo" by Songyou Peng, Bjoern Haefner, Yvain Queau, Daniel
% Cremers.
%
% [z_out, rho_out, N_out, iter, T_total] = SRPS(I, z0, mask, K, lambda, fill_missing_z, apply_smooth_filter, tol, max_iter, flag_cmg)
%
% INPUT:
% I - RGB image sequence 
% z0 - input low-resolution depth sequence 
% mask - binary mask for the object
% K - intrinsic matrix 
% lambda - tuning parameter for the photometric stereo term
% fill_missing_z - flag for inpainting the input depth (1 or 0)
% apply_smooth_filter - flag for smoothing the input depth (1 or 0)
% tol - relative energy stoping criterion 
% max_iter - maximum number of iteration
% flag_cmg - flag for CMG precondition (1 or 0)
%
% OUTPUT:
% z_out - super-resolution depth map
% e_rmse - root mean square error (RMSE)
% e_mae - mean angular error (MAE)
% iter - total number of iteration
% T_total - total runtime
%
% Author: Songyou Peng

addpath(strcat(pwd,'/Data_synthetic'))
scaling_factor = size(I,1) / size(z0,1); % Scaling factor
% load the downsampling matrix (K in the paper)
if scaling_factor == 1
    npix = size(z0,1)*size(z0,2);
    D = spdiags(ones(npix,1),0,npix,npix);
else 
    load(sprintf('D_%d_%d_%d.mat',size(I,2), size(I,1), scaling_factor));
end

%------------------------------------------------------
flag_synthetic =1;
if (flag_synthetic)
    z_ground = z;
    [nrows,ncols,nchannels] = size(N_ground);
    N_ground = reshape(N_ground,[nrows * ncols,nchannels]);    
end
%------------------------------------------------------

%%

% 
nb_harmo = 4; % Number of Spherical Harmonics parameters

% CG Parameters
flag_pcg = 1; % 1 for PCG, 0 for Least Square
maxit_cg = 100; % Max number of iterations for the inner CG iterations
tol_cg = 1e-9; % Relative energy stopping criterion for the inner CG iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing

% define the mask for the small depth
% masks = mask(1:scaling_factor:end,1:scaling_factor:end);
masks = D*mask(:);
masks = reshape(masks,size(z0,1),size(z0,2));
masks(masks<1) = 0;

% Pre-processing the input depth(s)

zs = z0;
zs(zs==0) = NaN;
zs = mean(zs,3);

if(fill_missing_z)
    disp('Inpainting');
    tic

    zs=inpaint_nans(zs);
    
    disp('Done');
    toc
end 

% Smoothing
if(apply_smooth_filter)

    disp('Filtering...');
    max_zs = max(max(zs.*masks));

    tic
    zs= imguidedfilter(zs/ max_zs);
    toc
    zs = zs.*max_zs;
    disp('Done')

end

% initial big depth map
z = imresize(mean(zs,3), scaling_factor,'bicubic');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some useful stuff

[nrows,ncols,nchannels, nImg] = size(I);

% pixel index inside the big and small mask
imask = find(mask>0);
imasks = find(masks>0);

% number of pixel
npix = length(imask);
npixs = length(imasks);

% new downsampling matrix, npixs * npix
KT= D(imasks, imask);

%__________________
T= KT'*masks(imasks);
T(T>0) = 1;
out = zeros(size(mask));
out(imask) = T;
mask = out;
imask = find(mask>0);
npix = length(imask);
KT= D(imasks, imask);


% T= KT'*masks(imasks);
% T(T>0) = 1;
% 
% out = zeros(size(mask));
% out(imask) = T;
% mask = out;
% imask = find(mask>0);
% npix = length(imask);
% KT= D(imasks, imask);
% iboundry = KT * mask(imask);
% iboundry(iboundry<1) = 0;
% M = spdiags(iboundry, 0, npixs, npixs);


% For building surface normals
[xx,yy] = meshgrid(1:ncols,1:nrows);
xx = xx(imask);
xx = xx-K(1,3);
yy = yy(imask);
yy = yy-K(2,3);

G = make_gradient(mask); % Finite differences stencils
Dx = G(1:2:end-1,:);
Dy = G(2:2:end,:);
clear G

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization

% Initial guess - lighting
s = zeros(size(I,3),9, size(I,4));s(:,3, :) = -1; % frontal directional lighting 

% Initial guess - albedo
rho = 0.5.*ones(size(I(:,:,:,1)));

% Vectorization
I = reshape(I,[nrows * ncols,nchannels, nImg]);
I = I(imask,:, :); % Vectorized intensities

rho = reshape(rho,[nrows * ncols,nchannels]);
rho = rho(imask,:); % Vectorized albedo

z = z(imask);
z0s = zs(imasks);


% Initial gradient
zx = Dx*z;
zy = Dy*z;

% Initial augmented normals
N = zeros(npix,nb_harmo); 
dz = max(eps,sqrt((K(1,1)*zx).^2+(K(2,2)*zy).^2+(-z-xx.*zx-yy.*zy).^2));
N(:,1) = K(1,1)*zx./dz;
N(:,2) = K(2,2)*zy./dz;
N(:,3) = (-z-xx.*zx-yy.*zy)./dz;
N(:,4) = 1;

E = [NaN];
iter = 1;

T_total = 0;
if do_display
    disp('Starting algorithm');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while 1
	%% lighting estimation
	t_start = tic;
    for i = 1 : nImg 
        for ch = 1:nchannels
            A = bsxfun(@times,rho(:,ch),N(:,1:nb_harmo));
            b = I(:,ch, i);

            s_now = solve_framework(s(ch,1:nb_harmo, i)', sparse(A), b,...
                                    flag_pcg, tol_cg, maxit_cg);

            s(ch,1:nb_harmo, i) = transpose(s_now); 
        end
    end
	s(:,nb_harmo+1:end, :) = 0;

	t_light_cur = toc(t_start);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Estimate rho
	t_start = tic;	
	for ch = 1 : nchannels
		A = N(:,1:nb_harmo)*squeeze(s(ch,1:nb_harmo,:)); % NPIX x NIMGS
		A = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A(:));
		b = I(:,ch,:);
		b = b(:);
		rho(:, ch) = solve_framework(rho(:,ch), A, b,...
							         flag_pcg, tol_cg, maxit_cg);
	end
	t_albedo_cur = toc(t_start);

	% Plot estimated albedo
    if do_display

        out = zeros(nrows, ncols);
        rho_plot = ones(nrows, ncols,3);
        out(imask) = min(rho(:,1), median(rho(:,1))+ 5 * std(rho(:,1)));
        rho_plot(:,:,1) = out;
        out(imask) = min(rho(:,2), median(rho(:,2))+ 5 .* std(rho(:,2)));
        rho_plot(:,:,2) = out;
        out(imask) = min(rho(:,3), median(rho(:,3))+ 5 * std(rho(:,3)));
        rho_plot(:,:,3) = out;
        figure(2);imagesc(max(0,min(1,rho_plot)));title('estimated albedo');
        axis off;
        drawnow
    end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Depth refinement

	A = [];
	B = [];
	t_start = tic;

	for ch = 1:nchannels

		B_ch = squeeze(I(:,ch,:)) ...
               - bsxfun(@times,rho(:,ch),N(:,4)*squeeze(s(ch,4,:))');
        
		
		B = [B; B_ch(:)];

		A_ch_1 = bsxfun(@times,rho(:,ch)./dz,bsxfun(@minus,transpose(K(1,1)*squeeze(s(ch,1,:))),bsxfun(@times,xx,transpose(squeeze(s(ch,3,:))))));
		A_ch_1 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_1(:));

		A_ch_2 = bsxfun(@times,rho(:,ch)./dz,bsxfun(@minus,transpose(K(2,2)*squeeze(s(ch,2,:))),bsxfun(@times,yy,transpose(squeeze(s(ch,3,:))))));
		A_ch_2 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_2(:));
		
        A_ch_3 = bsxfun(@times,rho(:,ch)./dz,transpose(squeeze(s(ch,3,:))));
        A_ch_3 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_3(:));
        
		A_ch = A_ch_1*Dx+A_ch_2*Dy - A_ch_3;

		A = [A;A_ch];
	end
	

    A_ = KT' * KT + lambda .* (A' * A);
	B_ = KT' * z0s + lambda .* (A' * B);
    
    
	z = solve_framework(z, A_, B_,...
                        flag_pcg, tol_cg, maxit_cg, flag_cmg);
    
    z_new = z;

	t_depth_cur = toc(t_start);                    
	

	% Energy
	E = [E, sum((KT*z - z0s).^2) + lambda * sum((A * z - B).^2)];

	% Relative residual
	rel_res = abs(E(end)-E(end-1))./abs(E(end));
	 
	% Update Normal map
	zx = Dx * z;
	zy = Dy * z;
    dz = max(eps,sqrt((K(1,1)*zx).^2+(K(2,2)*zy).^2+(-z-xx.*zx-yy.*zy).^2));
    N(:,1) = K(1,1)*zx./dz;
    N(:,2) = K(2,2)*zy./dz;
    N(:,3) = (-z-xx.*zx-yy.*zy)./dz;
    N(:,4) = 1;

    
    % Runtime 
    T_total = T_total + t_light_cur+t_albedo_cur+t_depth_cur;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Plot and Prints
	

if do_display    
    fprintf('[%d] light : %.2f, albedo : %.2f, depth: %.2f, total time: %.2f\n',...
			iter, t_light_cur, t_albedo_cur, t_depth_cur, t_light_cur+t_albedo_cur+t_depth_cur);
	fprintf('[%d] E : %.2f, R : %f\n',...
			iter, E(end), rel_res);		
    
    figure(6); plot(E);title('Energy');
	% visualize depth
	figure(3);
	out = zeros(nrows, ncols);
	outs = zeros(size(masks));
    
    outs(imasks) = z0s;
    subplot('Position', [0.05, 0.02, 0.6/scaling_factor, 0.8/scaling_factor]);
    visualize_depth(outs,masks);title(sprintf('input depth (%d * %d)',nrows/scaling_factor,ncols/scaling_factor));
    
    out(imask) = z_new;
    subplot('Position', [0.4, 0.02, 0.6, 0.8]);
    visualize_depth(out, mask); title(sprintf('refined SR depth (%d * %d)',nrows,ncols));
	drawnow;
    
	% Plot Normal map
    figure(5);
    Ndx = zeros(size(mask));Ndx(imask) = N(:,1);
    Ndy = zeros(size(mask));Ndy(imask) = N(:,2);
    Ndz = zeros(size(mask));Ndz(imask) = -N(:,3);
    Nd = 0.5*(1+cat(3,Ndx,Ndy,Ndz));

    imagesc(min(1,max(0,Nd))); title('Refined Normal Map')
    axis off
    drawnow

end

%------------------------------------------------------
if(flag_synthetic)
    % RMSE
    e_rmse = sqrt(sum((z_ground(imask)- z_new).^2)/numel(z_ground(imask)));

    % Normal difference
    AE_map = rad2deg(acos(sum(N(:,1:3) .* N_ground(imask,:),2)));
    e_mae = mean(AE_map);
    if do_display
        fprintf('[%d] RMSE: %f,  MAE: %f\n\n', iter, e_rmse, e_mae);
    end

end
%------------------------------------------------------    
iter = iter + 1;


	% Test CV
	if(rel_res<tol || iter>max_iter || E(end) > E(end-1))
        z_out = out;
        z_out(z_out==0) = NaN;    
		break;
    end
end

