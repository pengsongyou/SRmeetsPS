function [ D, crows, ccols ] = getDownsampleMat( scale, srrows, srcols )
%getDownsampleMat computes the matrix D that downsample an image by a
%factor scale
%INPUT:
%   scale > 1 is the factor that the sr image will be shrinked by, e.g.
%   scale=2 means the sr image will be compressed by a factor of 2, the
%   resulting image is half the size.
%   hx is the length each pixel has in x direction in the sr image
%   hy is the length each pixel has in y direction in the sr image
%OUTPUT:
%   D is the output matrix of size crows*ccols x srrows*srcols
%   crows is the number of rows of the resulting image
%   ccols is the number of cols of the resulting image
%
% Author: Bjoern Haefner

hx = 1; hy = 1;

% compute pixel size of coarse image (scale>1)
shx = scale*hx;
shy = scale*hy;

%resulting number of rows and columns in coarse image.
crows = floor(round(srrows/scale,9));
ccols = floor(round(srcols/scale,9));

%rows and cols positions in super resolution image wrt hx & hy
%Srows = [1:srrows]*hy;
%Scols = [1:srcols]*hx;


%rows and cols positions in coarse image wrt hx & hy
srows = (1:crows)*shy;
scols = (1:ccols)*shx;

%remainder to check how much of the area of the last pixel in the sr image
%is affected
rowremainder = mod(srows,hy);
colremainder = mod(scols,hx);

%get last row/col of sr image that is fully transformed to coarse image
higherrowpos = (srows-rowremainder)/hy;
rightcolpos = (scols-colremainder)/hx;

%get position of row/cols in sr image that are splitted into different pixel
%in coarse image
fractionalrowcells = higherrowpos + 1;
fractionalcolcells  = rightcolpos +1;

% get percentage of how a row/col in sr image is split (left/lower part)
fractionalrowsize = rowremainder/hy;
fractionalcolsize = colremainder/hx;

%get first row/col number in sr image that is used for new pixel in coarse
%image
lowerrowpos = [0, higherrowpos(1:end-1)]+1;
leftcolpos = [0, rightcolpos(1:end-1)]+1;

%get number of rows/cols that should be considered when transforming sr to
%coarse image, e.g. entry 1 has value 3 means first row/col pixel of coarse
%image has 3 pixel in sr image that should be considered.
nofrows = higherrowpos - [0, higherrowpos(1:end-1)] + min(1,ceil(mod(rowremainder, shy)));
nofcols = rightcolpos - [0, rightcolpos(1:end-1)] + min(1,ceil(mod(colremainder, shx)));

%rowcounter to follow which row of D we're currently on
coarserow = 1;

%compute the number of nonzero elements that fill the sparse matrix D.
numberofelements = sum(kron(nofrows,nofcols));

%allocate vector for creating sparse matrix to safe time
rowvec = zeros(1,numberofelements);
colvec = zeros(1,numberofelements);
valuevec = zeros(1,numberofelements);

%last element that was added to the above three vectors rowvec, colvec,
%valuevec.
lastelement = 0;


%check for every new coarse image pixel which of the sr pixels have to be
%considered and store them in a vector weighted with their area they
%participate in the new coarse pixel
for colind = 1:length(leftcolpos)
    for rowind = 1:length(lowerrowpos)
        
        %one row of matrix D, allocation.
        location = zeros(1,srrows*srcols);
        
        %row and col of sr image that will be considered as starting
        %pixel now
        currentstartingrow = lowerrowpos(rowind);
        currentstartingcol = leftcolpos(colind);
        
        %number of sr cells that lie in current coarse pixel
        nofcurrentrow = nofrows(rowind);
        nofcurrentcol = nofcols(colind);
        
        %get positions of sr pixel that are considered for current pixel in coarse
        %image and get a linear index.
        [X,Y] = meshgrid(currentstartingrow:currentstartingrow+nofcurrentrow-1, currentstartingcol:currentstartingcol+nofcurrentcol-1);
        index = sub2ind([srrows,srcols], transpose(X(:)), transpose(Y(:)));
        
        %initially weight every element 100%
        location(index) = hx*hy;
        
        %get last row and col of sr image that does participate to new
        %coarse pixel
        currentendingrow = fractionalrowcells(rowind);
        currentendingcol = fractionalcolcells(colind);
        
        % get percentage of associated lower row and left col 
        partofstartingrow = (currentstartingrow == fractionalrowcells) .* (1-fractionalrowsize);
        partofstartingcol = (currentstartingcol == fractionalcolcells) .* (1-fractionalcolsize);
        
        %get row and col number of partially associated lower row and left col.
        numberofpartialstartingrow = fractionalrowcells(partofstartingrow~=0);
        numberofpartialstartingcol = fractionalcolcells(partofstartingcol~=0);
        
        % get percentage of associated upper row and right col 
        partofendingrow = (currentendingrow == fractionalrowcells) .* fractionalrowsize;
        partofendingcol = (currentendingcol == fractionalcolcells) .* fractionalcolsize;
        
        %get row and col number of partially associated upper row and right col.
        numberofpartialendingrow = fractionalrowcells(partofendingrow~=0);
        numberofpartialendingcol = fractionalcolcells(partofendingcol~=0);
        
        if ~isempty(numberofpartialstartingrow) %enters if lower row is partially in coarse pixel changes the value in location vector to it's corresponding ratio
        upperpartialindex = sub2ind([srrows,srcols], repmat(numberofpartialstartingrow,size(Y,1),1), Y(:,1));
        location(upperpartialindex) = location(upperpartialindex)*partofstartingrow(partofstartingrow~=0);
        end
        
        if ~isempty(numberofpartialstartingcol)%enters if left col is partially in coarse pixel changes the value in location vector to it's corresponding ratio
        leftpartialindex = sub2ind([srrows,srcols], X(1,:), repmat(numberofpartialstartingcol,1,size(X,2)));
        location(leftpartialindex) = location(leftpartialindex)*partofstartingcol(partofstartingcol~=0);
        end
        
        if ~isempty(numberofpartialendingrow)%enters if upper row is partially in coarse pixel changes the value in location vector to it's corresponding ratio
        lowerpartialindex = sub2ind([srrows,srcols], repmat(numberofpartialendingrow,size(Y,1),1), Y(:,1));
        location(lowerpartialindex) = location(lowerpartialindex)*partofendingrow(partofendingrow~=0);
        end
        
        if ~isempty(numberofpartialendingcol)%enters if right col is partially in coarse pixel changes the value in location vector to it's corresponding ratio
        rightpartialindex = sub2ind([srrows,srcols], X(1,:), repmat(numberofpartialendingcol,1,size(X,2)));
        location(rightpartialindex) = location(rightpartialindex)*partofendingcol(partofendingcol~=0);
        end
        
        %stores the values in vectors:
        %rowvec is the vector representing the current row of D;
        %colvec is the vector representing the columns of D that
        %participate in corresponding pixel (given by the current row) of
        %coarse image;
        %valuevec are the weighted (by their participating area) values the
        %sr pixel participiate in current coarse pixel.
        rowvec(lastelement+1:length(index)+lastelement) = coarserow*ones(1,length(index));
        colvec(lastelement+1:length(index)+lastelement) = index;
        valuevec(lastelement+1:length(index)+lastelement) = location(index);
        
        %update counter variables
        lastelement = length(index)+lastelement;
        coarserow = coarserow + 1;
        
    end %end loop along rows
end %end loop along cols

%create sparse matrix that will be the output
D = sparse(rowvec, colvec, valuevec, crows*ccols, srrows*srcols);

%normalize the sparse matrix by dividing with new pixel size.
D = D/(shx*shy);

end

