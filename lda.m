function [alpha,rsamples] = lda(samples,labels,rr)
% Linear discriminant analysis
% Usage:
% [alpha,rsamples] = lda(samples,labels,rr);
% Input:
%   samples: n x d matrix, each row is a sample.
%   labels: n x 1 column vector, labels of samples.
%   rr: number of discriminant vector.
% Output:
%   alpha: n x r matrix, each column denotes a discriminant vector.
%   rsamples: n x r matrix, new samples after projection.
%
% Author: Liefeng bo
% School of electronic engineering, Xidian University
% April, 2006

[n,d] = size(samples);
labelsnum = unique(labels);
c = length(labelsnum);
dim = min(c-1,d);
% dim=d;

tmean = mean(samples);
bsamples = samples - repmat(tmean,n,1);
tscatters = 0;
for i = 1:n
    tscatters = tscatters + (bsamples(i,:))'*bsamples(i,:);
end
tscatters = (tscatters + tscatters')/2;

r = rank(tscatters);

% rank is not smaller than dimensionality of samples
if n >= d & r == d;
    bscatters = zeros(d,d);
    for i = 1:c
        cindex{i} = find(labels == labelsnum(i));
        tmp = mean(samples(cindex{i},:)) - tmean;
        bscatters = bscatters + length(cindex{i})*tmp'*tmp;
    end
    bscatters = (bscatters + bscatters')/2;

    options.disp = 0;
    [alpha,variance] = eigs(bscatters,tscatters,dim,'lm',options);
    %  normalize the coeficients
    for i = 1:dim
        alpha(:,i) = alpha(:,i)/norm(alpha(:,i));
    end
    rsamples = samples*alpha;
    
% rank is smaller than dimensionality of samples
else
    bscatters = zeros(d,d);
    for i = 1:c
        cindex{i} = find(labels == labelsnum(i));
        tmp = mean(samples(cindex{i},:)) - tmean;
        bscatters = bscatters + length(cindex{i})*tmp'*tmp;
    end
    tscatters = tscatters + 0*eye(d);
    [U,variance,pc] = svd(tscatters);
    if nargin == 3
        r = rr;
    end
    P = pc(:,1:r);  
    options.disp = 0;
    [dc,variance] = eigs(P'*bscatters*P,variance(1:r,1:r),dim,'lm',options);
    alpha = P*dc;
    
%  normalize the coeficients
    for i = 1:dim
        alpha(:,i) = alpha(:,i)/norm(alpha(:,i));
    end
    rsamples = samples*alpha;
end
