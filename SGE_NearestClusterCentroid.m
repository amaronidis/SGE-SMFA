function [z,MinDist,C] = SGE_NearestClusterCentroid(X,y,U)

%NEAREST CLUSTER CENTROID
%
%Inputs ->          X: Train Set
%                   y: Train Labels
%                   U: Test Set
%
%Outputs->          z: Estimated Test Labels
%                   MinDist: The minimun distances
%                   C: Cluster Centroids per Class


[M,N] = size(U);

NumOfClasses = max(y(1,:));

ClustersPerClass = zeros(1,NumOfClasses);

C = cell(1,NumOfClasses);

for i=1:NumOfClasses;
ClustersPerClass(i) = max(y(2,y(1,:)==i));
C{i} = zeros(M,ClustersPerClass(i));
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
        
        C{i}(:,j) = mean(Cluster,2);
        
    end
end

ClusterDistances = cell(1,NumOfClasses);

%Distances of diverse points to every class
D = zeros(NumOfClasses,N);

%Ids of minimum distant clusters per point
IdDistances = zeros(NumOfClasses,N);

for i=1:NumOfClasses
    
    %Distances of diverse points to every cluster of specific class
    ClusterDistances{i} = zeros(ClustersPerClass(i),N);
    
    for p=1:N
        for j=1:ClustersPerClass(i)
            ClusterDistances{i}(j,p) = sqrt(sum((U(:,p)-C{i}(:,j)).^2)); 
        end
    end
    
    if(size(ClusterDistances{i},1)==1)
        %In this case cluster = class
        D(i,:) = ClusterDistances{i};
        IdDistances(i,:) = ones(1,size(ClusterDistances{i},2));
    else
        [minClusterDist,ClusterId] = min(ClusterDistances{i});
        D(i,:) = minClusterDist;
        IdDistances(i,:) = ClusterId;
    end
    
    
end
        
[MinDist,z] = min(D);        