function [img, K, width, height] = loadData(img_size, path2files)
%INPUT:
%   img_size = [width height]
%   path2files, e.g. 'depths/'
%
%OUTPUT:
%   img is the rendered depth image
%   K is the corresponding intrinsic matrix which was used for rendering
%
%Example:
%   [img, K] = loadData([960 1280], 'depths/');
%   imshow(img, [], 'Colormap', hot);
%   disp(num2str(K));
%
% Author: Bjoern Baefner 

%load image
fileID = fopen([path2files,'joy',num2str(img_size(1)),'x',num2str(img_size(2)),'.bin']);
img = transpose(fread(fileID,img_size, 'float'));
fclose(fileID);

%load intrinsics
K = dlmread([path2files,'K',num2str(img_size(1)),'x',num2str(img_size(2)), '.txt']);

width = K(1,1);
height = K(1,2);
K = K(2:end,1:end);

end