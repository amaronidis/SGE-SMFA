function [XTrain,XTest] = SGE_CrossDivision(X,type,TrainIds,TestIds)

%CROSS DIVISION

%This function divides the given set into 2 subsets: The training set and the test set
%The test set consists of the 1/nFold percent of the whole set and
%the train set consists of the rest.
%
%Inputs->               X:          The Data Matrix (M x N, where M is the dimensionality and N is the number of the samples)
%                       type:       'data' for data matrix
%                                   'gram' for gram matrix
%                       TrainIds:   As it says
%                       TestIds:    As it says
%
%Outputs->              XTrain:     The training set
%                       XTest:      The testing set

if(strcmp(type,'data')==1)

    XTrain = X(:,TrainIds);
    XTest = X(:,TestIds);

elseif(strcmp(type,'gram')==1)
    
    XTrain = X(TrainIds,TrainIds);
    XTest = X(TrainIds,TestIds); 
    
end