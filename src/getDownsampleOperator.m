function [ D ] = getDownsampleOperator( size_from, size_to )
%getDownsampleOperator computes the matrix D that samples an image by a scale
%determined by size_from./size_to. Note that it accepts different scaling
%factors in different dimensions.
%INPUT:
%   size_from is a vector describing the size of the "input" data
%   size_to is a vector describing the size of the "output" data
%   Note: Only the first two values of size_{to,from} are taken into
%   account. If size_{to,from} is a scalar, then second dimension is
%   assumed to be 1.
%OUTPUT:
%   D is the output matrix of size (size_to(1)*size_to(2)) x (size_from(1)*size_from(2))
%   Note: D  acts like a 'box' operator.
%
% Copyright by
% Author: Bjoern Haefner
% Date: March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

rows_from = size_from(1);
if size(size_from,2)>=2
  cols_from = size_from(2);
else
  cols_from = 1;
end

rows_to = size_to(1);
if size(size_to,2)>=2
  cols_to = size_to(2);
else
  cols_to = 1;
end


if isequal([rows_from,cols_from],[rows_to,cols_to])
  pixels = rows_from*cols_from;
  D = spdiags(ones(pixels,1),0,pixels,pixels);
  return;
end

scale_y = rows_from/rows_to;
scale_x = cols_from/cols_to;

rows_frac = getFracMat(rows_to,rows_from, scale_y);
cols_frac = getFracMat(cols_to,cols_from, scale_x);

D = getResizeMat(rows_frac,cols_frac);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%getFracMat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [frac_mat] = getFracMat(dims_to,dims_from, scale)

frac_mat = zeros(dims_to,dims_from);
dim_to = 1;
dim_from = 1;
val = min(1,scale);
while (true)
  
  if (dim_to>size(frac_mat,1) || dim_from>size(frac_mat,2))
    break;
  end
  
  frac_mat(dim_to,dim_from) = val;
  
  dim_from_sum = sum(frac_mat(dim_to,:));
  dim_to_sum = sum(frac_mat(:,dim_from));
  
  if (dim_from_sum >= scale && dim_to_sum >= 1)
    dim_to = dim_to + 1;
    dim_from = dim_from + 1;
    val = min(1,scale);
    continue;
  end
  
  if dim_from_sum >= scale
    dim_to = dim_to + 1;
    val = min(1-dim_to_sum,scale);
  end
  
  if dim_to_sum >= 1
    dim_from = dim_from + 1;
    val = min(scale-dim_from_sum,1);
  end
  
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%getResizeMat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D] = getResizeMat(rows_frac,cols_frac)

[rows_to, rows_from] = find(rows_frac);
[cols_to, cols_from] = find(cols_frac);

[xx_from,yy_from] = meshgrid(cols_from,rows_from);
lin_from = (xx_from-1)*size(rows_frac,2) + yy_from;

[xx_vals,yy_vals] = meshgrid(cols_frac(cols_frac~=0),rows_frac(rows_frac~=0));
vals = xx_vals.*yy_vals;

[xx_to,yy_to] = meshgrid(cols_to,rows_to);
lin_to = (xx_to-1)*size(rows_frac,1) + yy_to;

D = sparse(lin_to(:),lin_from(:),vals(:),size(rows_frac,1)*size(cols_frac,1),size(rows_frac,2)*size(cols_frac,2));

%normalize
W = spdiags(1./sum(D,2),0,size(rows_frac,1)*size(cols_frac,1),size(rows_frac,1)*size(cols_frac,1));
D = W*D;

end