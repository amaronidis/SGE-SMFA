function [ConfMatrix,ConfMatrixIds] = SGE_ConfusionMatrix(yEst,y)

%CONFUSION MATRIX

%This function calculates the confusion matrix given the correct and the
%predicted labels

%Inputs->       y: Row vector with old labels
%               yEst: Row vector with new labels
%
%Outputs->      ConfMatrix:     Confusion Matrix
%               ConfIdsMatrix:  A cell matrix on place (i,j) of which there are the 
%                               ids of old samples of class i that go to
%                               class j

NumOfClasses = max(y);

ConfMatrix    = zeros(NumOfClasses,NumOfClasses);
ConfMatrixIds = cell(NumOfClasses,NumOfClasses);

for i=1:NumOfClasses
    
    for j=1:NumOfClasses
        
        OLD = find(y==i);
        Ids = OLD(yEst(OLD)==j);
        ConfMatrix(i,j) = length(Ids);
        ConfMatrixIds{i,j} = Ids;
        
    end
    
end