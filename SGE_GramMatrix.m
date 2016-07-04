function S = SGE_GramMatrix(varargin)

%GRAM MATRIX

%This function finds the Similarity matrix between two data matrices A and B
%
%Input->        A:      1st Data matrix (MA x NA, where MA is the dimensionality and NA is the number of samples)
%               B:      2nd Data matrix (MB x NB, where MB is the dimensionality and NB is the number of samples)
%               metric: 'euc' for Euclidean Similarity (1 -||u-v||/max||*||            
%                       'cor' for Correlation Similarity (<u,v>/|u||v|)
%                       'gau' for Gaussian Similarity (exp(-||u-v||^2/(2*sigma^2)
%               sigma:  Variance in case of Gaussian Similarity
%
%Output->       S:  The Similarity matrix
%
%Syntax:
%
%S = Gram_Matrix(A,metric)           metric: 'euc' or 'cor' 
%S = Gram_Matrix(A,metric,sigma)     metric: 'gau',  sigma: variance 
%S = Gram_Matrix(A,B,metric)         metric: 'euc' or 'cor'
%S = Gram_Matrix(A,B,metric,sigma)   metric: 'gau',  sigma: variance 

%Checking inputs
if (nargin<2)
    
    error('Not enough input arguments')
    

elseif((nargin)==2)
        
    if(ischar(varargin{2})~=1)
        error('No metric defined')
    elseif(strcmp(varargin{2},'euc')~=1 && strcmp(varargin{2},'cor')~=1 && strcmp(varargin{2},'gau')~=1)
        error('%s is not a valid metric',varargin{2})
    elseif(strcmp(varargin{2},'gau')==1)
        error('sigma parameter is not defined')
    else
        A = varargin{1};
        metric = varargin{2};
    end

    
    NA = size(A,2);

    S = zeros(NA,NA);

        switch metric
                
            case 'euc'
                    
                D = pdist2(A',A');
                max_dist = max(D(:));
                S = 1 - (D / max_dist);
                    
            case 'cor'
                    
                for i=1:NA
                    for j=(i+1):NA
                        S(i,j) = (A(:,i)' * A(:,j)) / (norm(A(:,i)') * norm(A(:,j)'));
                    end
                end
                
                S = S + S';

                S = S + diag(ones(1,NA));
                             
        end
        

elseif (nargin==3)
        
    if(ischar(varargin{2})~=1 && ischar(varargin{3})~=1)
        error('No metric defined')
    elseif(ischar(varargin{2})==1)
        if(strcmp(varargin{2},'gau')~=1)
            error('Too many input parameters')
        else
            A = varargin{1};
            metric = varargin{2};
            sigma = varargin{3};
            NA = size(A,2);
            S = zeros(NA,NA);
        end
    elseif(ischar(varargin{3})==1)
        if(strcmp(varargin{3},'gau')==1)
            error('sigma parameter is not defined')
        elseif(strcmp(varargin{3},'euc')~=1 && strcmp(varargin{3},'cor')~=1)
            error('%s is not a valid metric',varargin{3})
        else
            A = varargin{1};
            B = varargin{2};
            metric = varargin{3};
            [MA,NA] = size(A);
            [MB,NB] = size(B);
    
            if (MA~=MB)
                error('Vectors of two data sets have different dimensionalities')
            end
    
            S = zeros(NA,NB);
        end
    end
    
        
    switch metric
        
        case 'euc'
            
            D = zeros(NA,NB);
                        
            for i=1:NA
                for j=1:NB
                    d = sum((A(:,i) - B(:,j)).^2);
                    D(i,j) = d;
                end
            end
            
            S = 1 - D/max(D(:));
            
        case 'cor'
    
            for i=1:NA
                for j=1:NB
                    S(i,j) = (A(:,i)' * B(:,j)) / (norm(A(:,i)') * norm(B(:,j)'));
                end
            end
            
        case 'gau'
            
            for i=1:NA
                for j=1:NA
                    d = sum((A(:,i) - A(:,j)).^2);
                    S(i,j) = exp((-d)/(2*(sigma^2)));
                end
            end
            
    end
    
    
elseif(nargin==4)
    
    if(ischar(varargin{3})~=1)
        error('No metric defined')
    else
        A = varargin{1};
        B = varargin{2};
        sigma = varargin{4};
    end
    
    [MA,NA] = size(A);
    [MB,NB] = size(B);
    
    if (MA~=MB)
        error('Vectors of two data sets have different dimensionalities')
    end
    
    S = zeros(NA,NB);
    
    for i=1:NA
        for j=1:NB
            d = sum((A(:,i) - B(:,j)).^2);
            S(i,j) = exp((-d)/(2*(sigma^2)));
        end
    end
    
else
    
    error('Too many input arguments')
    
end