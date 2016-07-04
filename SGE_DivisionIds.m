function [TrainIds,TestIds] = SGE_DivisionIds(y,nFold,TestPiece)

%DIVISION IDS

%This function finds the ids for the training and the test set regarding
%the cross validation procedure
%
%Inputs->       y: The total labels
%               nFold: Number of folds for cross validation
%               TestPiece: Which piece will be the test set
%
%Outputs->      TrainIds: As it says
%               TestIds:  As it says

NumOfClasses = max(y);
TrainIds = [];
TestIds = [];

for i=1:NumOfClasses
    
    %Specify the class
    ClassIds = find(y==i);
    
    %Specific class cardinality
    N = length(ClassIds);
    
    %Cardinality of each group 
    TestSize = floor(N/nFold);
    
    %List that indicates which samples have been used for the test set (1 if used, 0 if not used)
    UsedIndicator = zeros(1,N);
    
    %Starting point of test_set
    s = (TestSize * (TestPiece - 1)) + 1;                 
    
    %Ending point of test_set
    t = TestSize * TestPiece;
    
    TestIndices = s:t;
    TestIds = [TestIds ClassIds(TestIndices)];
    
    %We update the used_indicator list
    UsedIndicator(TestIndices) = 1;
    
    %Train_set will contain the rest samples
    TrainIndices = (UsedIndicator == 0);
    TrainIds = [TrainIds ClassIds(TrainIndices)];
    
end