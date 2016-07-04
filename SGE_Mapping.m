function [TransMatrix,EigVals] = SGE_Mapping(X,Dim,L,Lp)

%MAPPING

%This function, given the desired new dimensionality or the
%percentage of desired retained energy, it performs the mapping 
%of the data into the low-dimensional space

%Inputs->       X:      Data Matrix
%               Dim:  	Desired number of eigenvalues to keep
%               L:      The intrinsic Laplacian
%               Lp:     The penalty Laplacian
%
%Outputs->      TransMatrix: Transformation Matrix
%               EigVals:     Eigenvalues which correspond to eigenvector-columns of TransMatrix

N = size(L,2);
    
for i=1:N
    if(L(i,i)<0.0000001)
        L(i,i)=0;
    end
end
for i=1:N
    if(Lp(i,i)<0.0000001)
        Lp(i,i)=0;
    end
end

[EigVecs,EigVals] = eig(X*L*X',X*Lp*X');

EigVals = diag(EigVals)';

%Check eigenvalues not to be nans
for p=1:length(EigVals)
    if(isnan(EigVals(p)))
        error('NaN eigenvalue')
    elseif(EigVals(p)<0.000001)
        EigVals(p) = 0;
    end
end

[EigVals,Ids] = sort(EigVals,'ascend'); 


if (Dim>=1)

    d = Dim;
    
elseif (Dim>0)
 
%The total sum of eigenvalues        
TotalEnergy = sum(EigVals);                           

%The sum we need to retain 
Threshold = Dim * TotalEnergy;                      
CurrentEnergy = 0;

for i=1:length(EigVals)
    CurrentEnergy = CurrentEnergy + EigVals(i);
    d = i;
    if (CurrentEnergy>=Threshold)
        break;
    end
end
      
end

TransMatrix = EigVecs(:,Ids(1:d));