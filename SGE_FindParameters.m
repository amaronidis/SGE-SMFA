function [delta,K] = SGE_FindParameters (M,EigVals)

%FIND PARAMETERS

%This function realizes relations (7) from the paper:
%'Spectral Methods For Automatic Multiscale Data Clustering'
%
%Inputs->               M: The power we lift the eigenvalues to
%                       EigVals: The eigenvalues of Laplacian
%
%Outputs->              delta: The maximum eigengap
%                       K: The corresponding K (see related paper)

%How many eigenvalues do we have
s = length(EigVals);                

%Row vector consists of the differences between consecutive eigenvalues^M
dif = zeros(1,s-1);                     

for i=1:(s-1)
    dif(i) = (EigVals(i))^M - (EigVals(i+1))^M;
end

%We use the maximal eigengap
[delta,K] = max(dif);                   