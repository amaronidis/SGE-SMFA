function [Plausible,M_max,DELTA,KAPPA] = SGE_FindPlausible(EigVals)

%FIND PLAUSIBLE

%This function finds the most plausible partitions of a data set given the eigenvalues of its Laplacian
%
%Inputs->       EigVals: The eigenvalues of Laplacian P
%
%Outputs->      Plausible: Struct with fields->
%                                   Kappa:      Row vector with all plausible K
%                                   Delta:      Row vector with all the corresponding delta
%                                   M:          Row vector with all the plausible M
%               M_max:     Maximum M before we get K=1
%               DELTA:     Several delta for every M
%               KAPA:      The corresponding K

%This row vector will have all delta for M=1:M_max
DELTA = [];     

%This row vector will have all the corresponding K according to relations (7)
KAPPA = [];                                             

M=1;

[delta,K] = FindParameters(M,EigVals);

DELTA = [DELTA delta];

KAPPA = [KAPPA K];

M = M + 1;

%We require that eigenvalues(K)<1 because there is an occasion where K>1
%but because eigenvalues(K)=1 and eigenvalues(K+1)<1 the difference of their power 
%will tend to one in a way that we will find no more local maxima of DELTA
while (K>1 && EigVals(K)<1)       
    
    [delta ,K] = FindParameters(M,EigVals);     
    
    DELTA = [DELTA delta];   
    
    KAPPA = [KAPPA K];
    
    M = M + 1;
    
end

if (K==1)
    
    %We don't want to partition the data into one class
    KAPPA(end) = [];                                     
    DELTA(end) = [];
    
else if (EigVals(K)==1)
        
        %Of course if eigenvalues(K)=1 and eigenvalues(K+1)<1,
        %then essentially the eigengap between the power of them remains maximum and will become 1
        DELTA(end) = 1;                                   
                                                        
    end
    
end

M_max = M-1;

if (isempty(DELTA)==0)

%If the set of delta is not empty, then we try to find the local maxima
%because they contain information for partition
[Plausible.Delta,Plausible.M] = FindLocalMax(DELTA);  
                                                        
Plausible.Kappa = KAPPA(Plausible.M);

else
    
    %If the set of delta is empty, there are no good partition anyway
    Plausible.Delta = [];
    Plausible.M = [];                                   
    Plausible.Kappa = [];

end