function [C,Cov] = SGE_ClassCentroids(X,y)

%CLASSCENTROIDS

%This function returns the centroids of the classes/subclasses of X. It
%also returns the covariance matrices of the above classes/subclasses.

%Inputs->   X:  Train Set, (M x N, where M is the dimensionality and N is the number of samples)
%           y:  Train Labels, (1 x N or 2 x N for class and subclass* labels where N as above)
%                       *First row class, second subclass
%
%Outputs->  C:  The centroids of the classes/subclasses in a cell array. In
%               the case of classes, each cell is a centroid. In the case of subclasses,
%               each cell is a matrix that contains the subclass centroids in columns
%           Cov:The inverse covariance matrices collected in a cell array, where
%               each cell corresponds to a specific class. Every cell consists
%               of cell arrays of the number of subclasses

[M,~] = size(X);

%The total number of classes
NumOfClasses = max(y(1,:));

C   = cell(1,NumOfClasses);
Cov = cell(1,NumOfClasses);

if(size(y,1)==1)
    
    %Firstly find the centroids of the several classes
    for i=1:NumOfClasses

        Class = X(:,y==i);

        C{1,i}   = mean(Class,2);
        Cov{1,i}{1,1} = cov(Class');

    end
    
else

    ClustersPerClass = zeros(1,NumOfClasses);
    
    for i=1:NumOfClasses;
        
        ClustersPerClass(i) = max(y(2,y(1,:)==i));
        C{1,i}   = zeros(M,ClustersPerClass(i));
        Cov{1,i} = cell(1,ClustersPerClass(i));
        
    end

    for i=1:NumOfClasses

        %Class CDA labels
        Idi = find(y(1,:)==i);
        yClass = y(:,Idi);
        Class = X(:,Idi);

        for j=1:ClustersPerClass(i)

            %Several clusters of specific class
            Idj = find(yClass(2,:)==j);
            Cluster = Class(:,Idj);

            C{1,i}(:,j) = mean(Cluster,2);
            Cov{1,i}{1,j} = cov(Cluster');

        end
        
    end    
    
end   