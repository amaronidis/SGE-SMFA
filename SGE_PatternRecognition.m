function Results = SGE_PatternRecognition(X,y,U,z,Dim,Par,k,Metric,pcaDim)

%PATTERN RECOGNITION

%This function performs training on a given training dataset, then maps a
%testing dataset according to the trained projection-eigenvectors. Finally,
%it classifies the mapped testing data and evaluates the accuracy rates.

%Inputs->       X:      Data Matrix for Linear case, (M x N, where M is the dimensionality and N is the number of samples)
%                       Gram Matrix for Kernel case, (N x N, where N as above)
%               y:      Data Labels, (1 x N in case of LPP, PCA, LDA)
%                                    (2 x N in case of CDA, SDA and SGE)
%               U:      Test Data Matrix for Linear case, (M_test x N_test)
%                       Test Gram Matrix for Kernel case, (N x N_test)
%               z:      Test labels, (1 x N_test)
%               Dim:    Desired dimensionality after projection
%               Par:    A struct with fields: 
%                                            - mode:    'LPP', 'PCA', 'LDA', 'MFA', 'CDA', 'SDA' or 'SGE'
%                                            - X:       The data matrix ('LPP' case)
%           `                                - sigma:   A number that multiplied by the mean distance between 
%                                                       the datapoints of X gives the variance of gaussian 
%                                                       for constructing the affinity matrix ('LPP' case)
%                                            - P:       The intrinsic mask ('MFA', 'SMFA' cases) 
%                                            - Q:       The penalty mask ('MFA', 'SMFA' cases)
%               k:      Classification mode
%               Metric: 
%                       - 'euc':    Euclidean,
%                       - 'mah1':   Mahalanobis which uses for each
%                                   class/subclass its specific covariance matrix
%                       - 'mah2':   Mahalanobis which uses the summation of
%                                   the covariance matrices of each class/subclass
%               pcaDim: The desired retained dimensionality after
%                       performing pca:
%                                       - pcaDim = 0: no PCA step
%                                       - 0<pcaDim<1: for retaining a percentage of the energy of the 
%                                                     covariance matrix of the initial data
%                                       - 1<pcaDim:   for retaining a specific number of dimensions
%
%Output->       Results: Struct with fields->
%                                           Mapped
%                                           TrainRates
%                                           TestRates

%optional PCA preprocessing step
if(pcaDim>0)
    [Xpca,PC] = pca(X,pcaDim);
    Upca = PC' * U;
else
    Xpca = X;
    Upca = U;
end

Mapped = SGE_DimReduction(Xpca,y,Upca,Dim,Par);

MappedX = Mapped.TrainSet;
MappedU = Mapped.TestSet;

[TrainRates,TestRates] = SGE_Assessment(MappedX,y,MappedU,z,k,Metric);


Results.Mapped = Mapped;
Results.TrainRates = TrainRates;
Results.TestRates = TestRates;