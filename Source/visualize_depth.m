function visualize_depth(z, mask)
% visualize the object shape
% 
% INPUT:
% z - depth map
% mask - binary mask
%
% Author: Yvain Queau

    z(mask==0) = NaN; % To not display outside the mask
    z(z==0) = NaN;
    h = surfl(-z,[0 90]); % Plot a shaded depth map, with frontal lighting
    axis equal; % Sets axis coordinates to be equal in x,y,z
    axis ij; % Uses Matlab matrix coordinates instead of standard xyz
    axis off; % Removes the axes
    shading flat; % Introduces shading
    colormap gray; % Use graylevel shading
    view(0,90) % Set camera to frontal
end