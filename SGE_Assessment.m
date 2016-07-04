function [TrainRates,TestRates] = SGE_Assessment(X,y,U,z,k,Metric)

%ASSESSMENT

%This function performs classification on a given test dataset and then extracts
%the confusion matrix in order to assess the classification performance rates.

%Inputs->       X:  Train Set, (M x N, where M is the dimensionality and N is the number of samples)
%               y:  Train Labels, (1 x N or 2 x N for class and subclass* labels where N as above)
%                       *First row class, second subclass
%               U:  Test Set, (MTest x NTest)
%               z:  Test Labels, (1 x NTest, where NTest as above)
%               k:  Classification Parameter->
%                                              (k>0) for KNN
%                                              (k=0) for Nearest Class Centroid
%                                              (k=-1) for Nearest Subclass Centroid
%                                              [a b c d e], for SVM (see osu-svm toolbox for parameters)
%               Metric: 'euc' : Euclidean
%                       'mah1': Mahalanobis which uses for each class/subclass its specific covariance matrix
%                       'mah2': Mahalanobis which uses the summation of the covariance matrices of each class/subclass
%
%
%Outputs->      TrainRates: Struct with fields->    
%                                               TotalRate: The rate evaluated on the whole data set
%                                               Precisions: Classification Rate for each class
%                                               MeanPrecision: Mean value of above class rates
%                                               Recalls: Recall for each class
%                                               MeanRecall: Mean recall of above class recalls
%                                               ConfMatrix: Confusion Matrix
%                                               ConfMatrixIds: Classified ids
%                                               EstLabels: Predicted Labels
%               TestRates: As above


%Classification Step
if(length(k)>1)
    %SVM takes labels of train and test data in a cell form
    ySVM{1} = y;
    ySVM{2} = z;
    [yEst,zEst] = SGE_Classification(X,ySVM,U,k);
else
    [yEst,zEst] = SGE_Classification(X,y,U,k,Metric);
end

    %Extracting Confusion Matrix
    [TrainConfMatrix,TrainConfMatrixIds] = SGE_ConfusionMatrix(yEst(1,:),y(1,:));
    [TestConfMatrix,TestConfMatrixIds]   = SGE_ConfusionMatrix(zEst(1,:),z);

    %Evaluating Rates
    [TrainPrecisions,TrainRecalls,TrainTotalRate] = SGE_Evaluation(TrainConfMatrix);
    [TestPrecisions,TestRecalls,TestTotalRate]    = SGE_Evaluation(TestConfMatrix);

    %Gathering Results for Train Set
    TrainRates.TotalRate     = TrainTotalRate;
    TrainRates.Precisions    = TrainPrecisions;
    TrainRates.MeanPrecision = sum(TrainPrecisions)/length(TrainPrecisions);
    TrainRates.Recalls       = TrainRecalls;
    TrainRates.MeanRecall    = sum(TrainRecalls)/length(TrainRecalls);
    TrainRates.ConfMatrix    = TrainConfMatrix;
    TrainRates.ConfMatrixIds = TrainConfMatrixIds;
    TrainRates.EstLabels     = yEst(1,:);
    TrainRates.CentDists     = yEst(2,:);

    %Gathering Results for Test Set
    TestRates.TotalRate      = TestTotalRate;
    TestRates.Precisions     = TestPrecisions;
    TestRates.MeanPrecision  = sum(TestPrecisions)/length(TestPrecisions);
    TestRates.Recalls        = TestRecalls;
    TestRates.MeanRecall     = sum(TestRecalls)/length(TestRecalls);
    TestRates.ConfMatrix     = TestConfMatrix;
    TestRates.ConfMatrixIds  = TestConfMatrixIds;
    TestRates.EstLabels      = zEst(1,:);
    TestRates.CentDists      = zEst(2,:);