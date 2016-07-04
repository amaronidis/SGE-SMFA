function [EigVecs,EigVals] = SGE_LaplacianEigenanalysis(W)

%LAPLACIAN EIGENANALYSIS

%This function finds the Random walk Laplacian of a Similarity Matrix and performs eigenanalysis on it
%
%Input->            W: Similarity Matrix
%
%Outputs->          EigVals: Row Vector with sorted (Descending) eigenvalues of Random Walk Laplacian P
%                   EigVecs: Matrix with columns the eigenvectors of Random Walk Laplacian P (also sorted)

%Diagonal matrix consists of the degrees of the vertices
D = diag(sum(W));                                   

%The Random Walk Laplacian
P = D\W;                                     

%Eigenanalysis of P
[EigVecs,EigVals] = eig(P);                             

s=size(EigVals,2);

%Transform matrix with eigenvalues to row vector
EigVals = diag(EigVals)';

for i=1:s  
    if (EigVals(i)>0.99999)
        EigVals(i) = 1; 
    end
end

%Sorted eigenvalues and their indices
[EigVals,Ids] = sort(EigVals,'descend');      

%We sort the eigenvectors according to their corresponding eigenvalues
EigVecs = EigVecs(:,Ids);                        