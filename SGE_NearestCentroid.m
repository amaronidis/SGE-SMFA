function [z,MinDist] = SGE_NearestCentroid(C,U,Par)

%NEAREST CENTROID

%This function implements the nearest centroid and the nearest cluster
%centroid classifiers.

%Inputs->           C: Centroids of the several classes
%                   U: The test set to be classified
%                   Par: A structure with fields:
%                                                 - Metric:
%                                                          * 'euc':    Euclidean,
%                                                          * 'mah1':   Mahalanobis which uses for each
%                                                                      class/subclass its specific covariance matrix
%                                                          * 'mah2':
%                                                          Mahalanobis which uses the summation of the specific covariances
%                                                  - Cov:  Contains the covariances of the classes/subclasses ('mah1' and 'mah2' case)                                                                      the covariance matrices of each class/subclass
%
%
%Outputs->          z: The estimated test labels
%                   MinDist: The minimun distances

[M,N] = size(U);

NumOfClasses = length(C);

D = zeros(NumOfClasses,N);
ClusterDistances = cell(1,NumOfClasses);

ClustersPerClass = zeros(1,NumOfClasses);

for i=1:NumOfClasses;
    
    ClustersPerClass(i) = size(C{1,i},2);

end

%Ids of minimum distant clusters per point
IdDistances = zeros(NumOfClasses,N);

if(strcmp(Par.Metric,'euc')==1) 

    for i=1:NumOfClasses

        %Distances of diverse points to every cluster of specific class
        ClusterDistances{1,i} = zeros(ClustersPerClass(i),N);

        for p=1:N

            for j=1:ClustersPerClass(i)

                ClusterDistances{1,i}(j,p) = sqrt(sum((U(:,p)-C{1,i}(:,j)).^2)); 

            end

        end

        if(size(ClusterDistances{1,i},1)==1)
            %In this case cluster = class
            D(i,:) = ClusterDistances{1,i};
            IdDistances(i,:) = ones(1,size(ClusterDistances{1,i},2));
        else
            [minClusterDist,ClusterId] = min(ClusterDistances{1,i});
            D(i,:) = minClusterDist;
            IdDistances(i,:) = ClusterId;
        end
        
    end

elseif(strcmp(Par.Metric,'mah1')==1)
    
    for i=1:NumOfClasses
        
        for j=1:ClustersPerClass(i)
            
            if(ClustersPerClass(i)==1)
                Par.Cov{1,i}{1,1} = inv(Par.Cov{1,i}{1,1});
            else
                Par.Cov{1,i}{1,j} = inv(Par.Cov{1,i}{1,j});
            end
            
        end
        
    end
                
    for i=1:NumOfClasses

        %Distances of diverse points to every cluster of specific class
        ClusterDistances{i} = zeros(ClustersPerClass(i),N);

        for p=1:N

            for j=1:ClustersPerClass(i)

                if(ClustersPerClass(i)==1)
                    ClusterDistances{i}(j,p) = sqrt((U(:,p)-C{1,i}(:,j))' * Par.Cov{1,i}{1,1} * (U(:,p)-C{1,i}(:,j))); 
                else
                    ClusterDistances{i}(j,p) = sqrt((U(:,p)-C{1,i}(:,j))' * Par.Cov{1,i}{1,j} * (U(:,p)-C{1,i}(:,j))); 
                end

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
    
elseif(strcmp(Par.Metric,'mah2')==1)
    
TotCov = TotalCovariance(M,Par);
TotCov = inv(TotCov);
    
    for i=1:NumOfClasses

        %Distances of diverse points to every cluster of specific class
        ClusterDistances{i} = zeros(ClustersPerClass(i),N);

        for p=1:N

            for j=1:ClustersPerClass(i)

                ClusterDistances{i}(j,p) = sqrt((U(:,p)-C{1,i}(:,j))' * TotCov * (U(:,p)-C{1,i}(:,j))); 

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
           
else
    
    error('Unknown metric');
    
end

[MinDist,z] = min(D);