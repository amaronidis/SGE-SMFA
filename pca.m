function [MappedX TransMatrix EigVals] = pca(X,energy)

%PCA

%This function performs PCA on a given Data Matrix 
%
%Inputs->       X:           The Data Matrix (M x N, where M is the dimensionality and N is the number of samples
%               energy:      The amount of energy we ask from the algorithm to retain. This must be a real number 
%                            between 0 and 1 if we determine the percentage of the total energy we need, 
%                            or a natural number if we need to retain a specific number of dimensions 
%
%Outputs->      MappedX:     The tranformed data into the lower dimensional space
%               TransMatrix: The Transformation Matrix which maps the data into the new feature space
%               EigVals:     Sorted keeped eigenvalues

%M:dimensionality  N:number of samples
[M N] = size(X);                                        

mn = mean(X,2);

%We first center the data
XCentered = X - repmat(mn,1,N);                        

%The covariance matrix
Covariance = 1 / (N-1) * XCentered * XCentered';      

%Eigenanalysis of cov matrix
[C, V] = eig(Covariance);                               

V = diag(V);

%We sort the eigenvalues in descending mode
[sortedV, Indices] = sort(V,'descend');                

if (energy>=1)

    p = energy;
    
elseif (energy>0)
 
%The total sum of eigenvalues        
totalEnergy = sum(sortedV);                           

%The sum we need to retain 
threshold = energy * totalEnergy;                      
currentEnergy = 0;

for i=1:N
    currentEnergy = currentEnergy + sortedV(i);
    p = i;
    if (currentEnergy>=threshold)
        break;
    end
end

else
    
    disp('energy parameter must be >0')
    return
        
end

%%The principal eigenvalues
EigVals = sortedV(Indices(1:p));                            

%The principal eigenvectors
TransMatrix = C(:,Indices(1:p));                                 

%Mapped data after projection
MappedX = TransMatrix' * X;                                     