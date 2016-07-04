function [Precisions,Recalls,TotalRate] = SGE_Evaluation(ConfMatrix)

%EVALUATION

%This function evaluates the performance of an algorithm 
%from the confusion matrix
%
%Inputs->       ConfMatrix: Confusion Matrix
%
%Outputs->      Precisions: Rates per class
%               Recalls:    Recalls per class
%               TotalRate:  Rate per whole set

numOfClasses = size(ConfMatrix,2);
ClassCards   = zeros(1,numOfClasses);
PredictCards = zeros(1,numOfClasses);
Precisions   = zeros(1,numOfClasses);
Recalls      = zeros(1,numOfClasses);
TotalRate    = 0;

for i=1:numOfClasses
    
    ClassCards(i)   = sum(ConfMatrix(i,:));
    PredictCards(i) = sum(ConfMatrix(:,i));
    
    if(ClassCards(i)==0)
        Precisions(i) = 0;
    else
        Precisions(i)   = ConfMatrix(i,i)/ClassCards(i);
        Precisions(i)   = Precisions(i) * 100;
    end
    
    if(PredictCards(i)==0)
        Recalls(i) = 0;
    else
        Recalls(i)      = ConfMatrix(i,i)/PredictCards(i);
        Recalls(i)      = Recalls(i) * 100;
    end

    TotalRate       = TotalRate + ConfMatrix(i,i);
    
end

TotalRate = TotalRate/sum(ClassCards);
TotalRate = TotalRate * 100;