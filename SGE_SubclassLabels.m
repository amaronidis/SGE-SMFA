function [yy,ClustersPerClass,TotalClusters] = SGE_SubclassLabels(ClustersOfClasses,y)

%SUBCLASS LABELS

%This function constructs the cluster labels of the samples for subclass-based algorithms.
%The labels constitute a 2 row matrix in the first row of which there are the
%class labels and in the second row there are the cluster labels of each sample.
%
%Inputs->               ClustersOfClasses: A cell, the output of the SubclassExtract algorithm
%                       y:                 The vector with the class labels of the samples
%
%Output->               yy:                The 2 row matrix with the whole information about the class labels and the subclass labels
%                       ClustersPerClass:  Number of subclasses in each class
%                       TotalClusters:     As it says

yy = zeros(2,length(y));
TotalClusters = 0;

%Of course the first row contains the initial class labels
yy(1,:) = y;                                           

%The total number of classes. We are going to work with every class separately
NumOfClasses = max(y);                                       

ClustersPerClass = zeros(1,NumOfClasses);

for i=1:NumOfClasses
    
    %We hold only the cluster-labels of this specific class
    ClustersOfSpecificClass = ClustersOfClasses{1,i}.Labels;   
    
    ClustersPerClass(i) = max(ClustersOfSpecificClass);
    TotalClusters = TotalClusters + ClustersPerClass(i);
    
    %We hold only the IDS of this specific class
    IDSOfSpecificClass = ClustersOfClasses{1,i}.InitClassIds;               
    
    ClassIds = find(y==i);
    
    for j=1:length(ClassIds)
    
    index = find(ClassIds(IDSOfSpecificClass)==ClassIds(j));
    
    %We gradually complete the second row of the CDA_labels matrix
    yy(2,ClassIds(j)) = ClustersOfSpecificClass(index); 
    
    end
    
end