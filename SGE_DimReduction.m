function Mapped = SGE_DimReduction(X,y,U,dim,Par)

%DIMENSIONALITY REDUCTION

%This function performs dimensionality reduction on a given dataset by utilizing a subspace method.

%Inputs->       X:      Data Matrix for Linear case, (M x N, where M is the dimensionality and N is the number of samples)
%                       Gram Matrix for Kernel case, (N x N, where N as above)
%               y:      Data Labels: (1 x N in case of LPP, PCA, LDA)
%                                    (2 x N in case of CDA, SDA and SGE)
%               U:      Test Data Matrix for Linear case, (M_test x N_test)
%                       Test Gram Matrix for Kernel case, (N x N_test)
%               dim:    Desired dimensionality after projection
%               Par:    A struct with fields: 
%                                            - mode:    'LPP', 'PCA', 'LDA', 'MFA', 'CDA', 'SDA' or 'SMFA'
%                                            - X:       The data matrix ('LPP' case)
%                                            - sigma:   A number that multiplied by the mean distance between 
%                                                       the datapoints of X gives the variance of gaussian 
%                                                       for constructing the affinity matrix ('LPP' case)
%                                            - P:       The intrinsic mask ('MFA', 'SMFA' cases) 
%                                            - Q:       The penalty mask ('MFA', 'SMFA' cases)
%
%Outputs->      Mapped: Struct with fields->
%                                           TrainSet:    Mapped Data Matrix
%                                           TestSet:     Mapped test Data Matrix
%                                           TransMatrix: Transformation Matrix
%                                           EigVals:     Eigenvalues sorted

[L,Lp] = SGE_GraphConstruct(y,Par);

[TransMatrix,EigVals] = SGE_Mapping(X,dim,L,Lp);

MappedX = Projection(X,0,TransMatrix);
MappedU = Projection(U,0,TransMatrix);

Mapped.TrainSet = MappedX;
Mapped.TestSet = MappedU;
Mapped.TransMatrix = TransMatrix;
Mapped.EigVals = EigVals;