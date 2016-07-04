function ClustersOfClasses = SGE_SubclassExtract(X,y,S,pt,ct)

%SUBCLASS EXTRACT

%This function extracts the several subclasses of each class of the data.
%It uses for every class:
%
%   a) the K-Means algorithm or 
%   b) the Multiple Spectral Clustering algorithm holding the most plausible partition results
%
%Inputs->       X:  Data Matrix (M x N, where M is the dimensionality and N is the number of samples)
%               y:  Data Labels (1 x N or 2 x N for class and subclass labels where N as above)
%               S:  Similarity Matrix (or Gram Matrix) for Spectral Clustering
%                   A (1 x NumOfClasses) vector with the number K of desired clusters per class for simple K-Means
%               pt: - Plausibility threshold for Spectral Clustering. The algorithm outputs only 
%                   partitions which have plausibility measure over pt 
%                   - 0 for simple K-Means
%               ct: - Cardinality threshold multiplier for SC: T = ct * (N/K), where T is the threshold, 
%                   ct the multiplier, N the number of samples and K the number of clusters 
%                   - 0 for simple K-Means
%
%Output->       ClustersOfClasses:   A (1 x NumOfClasses) cell array, which contains the whole information 
%                                    about the most plausible clustering in every class. 
%                                    1st field--> Partition: Most plausible partitions of every class
%                                    2nd field--> Labels:    Corresponding cluster-labeling
%                                    3rd field--> InitIds:   List with the initial in-class ids of the 
%                                                            corresponding samples

NumOfClasses = max(y);  

ClustersOfClasses = cell(1,NumOfClasses);

%K-Means case
if(size(S,1)==1)
    
    fprintf('\nExtracting subclasses using K-Means\n');
    
    for i=1:NumOfClasses
    
        ClassIds = find(y==i);
        XClass = X(:,ClassIds); 
        
        if(S(i)>1)
            InitialCenters = SGE_Maximin(XClass,S(i));

            Idx = kmeans(XClass',S(i),'maxiter',10000,'start',InitialCenters'); 

            [IdxBlock,ids] = sort(Idx);

            C   = X(:,ids); 
            yy  = IdxBlock';    
            ids = ids';
        else
            C   = XClass;
            yy  = ones(1,size(XClass,2));
            ids = 1:size(XClass,2);
        end
    
        ClustersOfClasses{1,i}.Partition    = C;                               
        ClustersOfClasses{1,i}.Labels       = yy;  
        ClustersOfClasses{1,i}.InitClassIds = ids;
        ClustersOfClasses{1,i}.InitIds      = ClassIds(ids);

        fprintf('Class %d complete\n',i');
            
    end
  
%Spectral Clustering case
else
    
    fprintf('\nExtracting subclasses using Multiple Spectral Clustering\n');

    for i=1:NumOfClasses

        fprintf('Class %d\n',i)

        ClassIds = find(y==i);
        
        %At every step we hold only the data samples of the specific class
        Xclass = X(:,ClassIds); 

        %At every step we hold only the within-class similarities
        Sclass = S(y==i,y==i);                                             

        %Here we apply the Multiscale_spectral_clustering algorithm    
        [Clusters,Measures] = SGE_MultiSpecCluster(Xclass,Sclass,pt,ct); 
%         fprintf('\n------------------------------------------------------------------')

        b = Measures.Plausibility;
        c = Measures.Cardinality;
        CardPassIds = find(c==1);
        b = b(CardPassIds);

        if(isempty(b)==1)

            ClustersOfClasses{1,i}.Partition    = Xclass;   
            ClustersOfClasses{1,i}.Labels       = ones(1,size(Xclass,2));
            ClustersOfClasses{1,i}.InitClassIds = 1:size(Xclass,2);
            ClustersOfClasses{1,i}.InitIds      = ClassIds(1:size(Xclass,2));

        else

        %We hold only the most plausible partition results
        [~,Index] = max(b); 

        %We hold only the most plausible results
        ClustersOfClasses{1,i}.Partition    = Clusters{CardPassIds(Index)}.Partitions;                               

        ClustersOfClasses{1,i}.Labels       = Clusters{CardPassIds(Index)}.Labels;

        ClustersOfClasses{1,i}.InitClassIds = Clusters{CardPassIds(Index)}.InitIds;
        
        ClustersOfClasses{1,i}.InitIds      = ClassIds(Clusters{CardPassIds(Index)}.InitIds);

        end

    end
    
%     fprintf('\n------------------------------------------------------------------\n')

end