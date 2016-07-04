function [yEst,zEst] = SGE_Classification(X,y,U,k,Metric)

%CLASSIFICATION

%This function performs training on a given set of training data and then
%classification on a given set of test data.

%Inputs->       X:	Train Set, (M x N, where M is the dimensionality and N is the number of samples)
%               y: 	Train Labels, (1 x N or 2 x N for class and cluster labels where N as above)
%                                 (SVM case: 1 X 2 cell with train and test Labels respectively)
%               U:  Test Set, (MTest x NTest)
%               k: 	Classification Parameter->
%                                         (>0) for KNN classification
%                                         (=0) for Nearest Class Centroid
%                                         (-1) for Nearest Subclass Centroid (CDA case)
%                                         [a b c d e], for SVM (see osu-svm toolbox for parameters)    
%               Metric: 'euc'  -> Euclidean
%                       'mah1' -> Mahalanobis that uses the specific covariance matrix of each class/cluster
%                       'mah2' -> Mahalanobis that uses the summation of the covariance matrices of the classes/clusters
%
%
%Outputs->      yEst: 1st row: Predicted Train Labels
%                     2nd row: Distances
%               zEst: 1st row: Predicted Test Labels
%                     2nd row: Distances

if(size(k,2)==1)
   
    if(k>0)
    
        %Classify the train samples
        yEst(1,:) = knnclassify(X',X',y(1,:),k,'euclidean');
        yEst(2,:) = nan(1,size(X,2));

        %Classify the test samples
        zEst(1,:) = knnclassify(U',X',y(1,:),k,'euclidean');
        zEst(2,:) = nan(1,size(U,2));

    elseif(k==0)

        %Find the centroids
        [C,Cov] = SGE_ClassCentroids(X,y);
        
        Par.Metric = Metric;
        Par.Cov    = Cov;
        
        %Classify the training samples
        [yEst(1,:),yEst(2,:)] = SGE_NearestCentroid(C,X,Par);       

        %Classify the testing samples
        [zEst(1,:),zEst(2,:)] = SGE_NearestCentroid(C,U,Par);
        
    end
    
else
    
        %SVMs
        [AlphaY,SVs,Bias,Parameters,nSV,nLabel] = SVMTrain(X,y{1},k);
        [~,~,~,~,yEst]= SVMTest(X, y{1}, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
        [~,~,~,~,zEst]= SVMTest(U, y{2}, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
        yEst(2,:) = nan(1,size(X,2));
        zEst(2,:) = nan(1,size(U,2));
        
end