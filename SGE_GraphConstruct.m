function [L,Lp] = SGE_GraphConstruct(y,Par)

%GRAPH CONSTRUCT

%This function constructs the intrinsic and penalty Laplacian graph matrix
%of an algorithm

%Inputs->       y:      Data Labels: (1 x N in case of LPP, PCA, LDA)
%                                    (2 x N in case of CDA, SDA and SGE)
%               Par:    A struct with fields: 
%                                            - mode:    'LPP', 'PCA', 'LDA', 'MFA', 'CDA', 'SDA' or 'SGE'
%                                            - X:       The data matrix ('LPP' case)
%           `                                - sigma:   A number that multiplied by the mean distance between 
%                                                       the datapoints of X gives the variance of gaussian 
%                                                       for constructing the affinity matrix ('LPP' case)
%                                            - P:       The intrinsic mask ('MFA', 'SMFA' cases) 
%                                            - Q:       The penalty mask ('MFA', 'SMFA' cases)
%
%Outputs->      L:      Intrinsic Graph Laplacian Matrix
%               Lp:     Penalty Graph Laplacian Matrix

switch Par.mode
    
    case 'LPP'
               
        %Number of samples
        N = size(y,2);
        
        X = Par.X;
        sigma = Par.sigma * mean(mean(pdist2(X',X')));
        
        %The intrinsic Laplacian matrix is essentially the similarity matrix
        W = GramMatrix(X,'gau',sigma);
        
        %The diagonal Degree matrix
        D  = diag(sum(W));
        
        L = D - W;
        
        %The penalty Laplacian matrix is the identity matrix
        Lp = eye(N,N);
    
    case 'PCA'
        
        %Number of samples
        N = size(y,2);
        
        %The intrinsic graph matrix
        L = eye(N,N);
        
        Wp = (1/N) * ones(N,N);
               
        Lp = eye(N,N) - Wp;
        
    case 'LDA'
        
        %Number of samples
        N = length(y);

        %The intrinsic graph matrix
        W = zeros(N,N);

        %Total number of classes
        NumOfClasses = max(y);

        for i=1:NumOfClasses

            ClassIds = find(y==i);

            ClassCard = length(ClassIds);

            W(ClassIds,ClassIds) = 1 / ClassCard;

        end
                
        L = eye(N,N) - W;
        
        Lp = W - (1/N)*ones(N,N);
        
    case 'MFA'
        
        S = Par.SimilMatrix;
        k = Par.kInt;
        kp = Par.kPen;
        
        %Intrinsic Graph Matrix
                
        %Number of samples
        N = length(y);

        %The number of classes
        NumOfClasses = max(y);                               

        %The intrinsic graph matrix
        W = zeros(N,N);                                             

        %We construct modified_S which is identical to S except for the diagonal elements,
        %where we put zeros so that we do not consider the identical points as neighboring points
        ModifiedS = S;

        for j=1:N      

            ModifiedS(j,j) = 0;

        end

        for i=1:NumOfClasses

            ClassIndices = find(y==i);

            %Number of samples belonging to this class
            nc = length(ClassIndices);                            

            IntraClassSimilarities = ModifiedS(ClassIndices,ClassIndices);

            [~,Ids] = sort(IntraClassSimilarities,'descend');

            %k might exceed the number of total neighbors inside a class
            p = min(nc - 1,k);

            for j=1:nc

            %If q belongs to the k nearest neighbors of w inside the class we put one
            W(ClassIndices(j),ClassIndices(Ids(1:p,j))) = 1;      

            end

        end

        for i=1:N

            for j=1:N

                %If i belongs to the k nearest neighbors of j or j belongs to the k nearest neighbors of i,
                %then we put an edge between the two nodes
                if(W(i,j)==1 || W(j,i)==1)      

                    W(i,j) = 1;   

                    W(j,i) = 1;

                end

            end

        end

        for s=1:N

            %We bring back ones on diagonal
            W(s,s)=1;  

        end
        


        %Penalty graph matrix
        Wp = zeros(N,N);                                                 

        for i=1:N

            %All indices of vertices which do not belong to the same class with ith vertex
            VertexSet = find(y(1,:)~=y(1,i));                       

            %Similarities between vertex i and vertices of VertexSet
            InterClassSimilarities = S(i,VertexSet);          

            [~,ids] = sort(InterClassSimilarities,'descend');

            %kp might exceed the number of outside neighbors
            p = min(length(VertexSet),kp);

            %If q belongs to the k nearest neighbors of w outside the class we put one
            Wp(i,VertexSet(ids(1:p))) = 1;                              

        end

        %We turn the undirected penaly graph into directed
        for i=1:N

            for j=1:N

                %If i belongs to the k nearest neighbors of j or j belongs to the k nearest neighbors of i,
                %we put an edge between the two nodes
                if(Wp(i,j)==1 || Wp(j,i)==1)    

                    Wp(i,j) = 1;  

                    Wp(j,i) = 1;

                end

            end

        end
        
        %The diagonal Degree matrix
        D  = diag(sum(W));      
        L = D - W;      
        
        %The diagonal Degree matrix
        D  = diag(sum(Wp));      
        Lp = D - Wp;            
        
        
    case 'SMFA'
        
        S = Par.SimilMatrix;
        k = Par.kInt;
        kp = Par.kPen;
        
        %Intrinsic Graph Matrix
                
        %Number of samples
        N = length(y);

        %The number of classes
        NumOfClasses = max(y(1,:)); 
        
        %List with the number of clusters belonging to each class
        NumOfClusters = zeros(1,NumOfClasses);

        %Cardinal numbers of clusters of several classes
        Cards = cell(1,NumOfClasses);

        %Initial ids of data points in several clusters of several classes
        IDS = cell(1,NumOfClasses);        

        %The intrinsic graph matrix
        W = zeros(N,N);                                             

        %We construct modified_S which is identical to S except for the diagonal elements,
        %where we put zeros so that we do not consider the identical points as neighboring points
        ModifiedS = S;

        for j=1:N      

            ModifiedS(j,j) = 0;

        end

        for i=1:NumOfClasses

%             ClassIndices = find(y==i);
            
            %The data points which belong to ith class
            ClassIds = find(y(1,:)==i);
            Class = y(2,ClassIds); 

            %The number of clusters in the ith class
            NumOfClusters(i) = max(Class);

            %For every cluster of ith class we have a cardinality
            Cards{i} = zeros(1,NumOfClusters(i));

            %For every class i, there is a list of ids of data points which belong to a specific cluster
            IDS{i} = cell(1,NumOfClusters(i));  
            
            for j=1:NumOfClusters(i)

                %The specific cluster of specific class data points initial ids
                IDS{i}{j} = ClassIds(Class==j);

                %The corresponding cardinalities
                Cards{i}(j) = length(IDS{i}{j});
                nc = Cards{i}(j);
                
                IntraClusterSimilarities = ModifiedS(IDS{i}{j},IDS{i}{j});
                
                [~,Ids] = sort(IntraClusterSimilarities,'descend');
                
                %k might exceed the number of total neighbors inside a
                %cluster
                p = min(nc - 1,k);

                for q=1:nc

                %If q belongs to the k nearest neighbors of w inside the cluster we put one
                W(IDS{i}{j}(q),IDS{i}{j}(Ids(1:p,q))) = 1;      

                end
                       
            end  
            
        end

        for i=1:N

            for j=1:N

                %If i belongs to the k nearest neighbors of j or j belongs to the k nearest neighbors of i,
                %then we put an edge between the two nodes
                if(W(i,j)==1 || W(j,i)==1)      

                    W(i,j) = 1;   

                    W(j,i) = 1;

                end

            end

        end

        for s=1:N

            %We bring back ones on diagonal
            W(s,s)=1;  

        end
        

        %Penalty graph matrix
        Wp = zeros(N,N);                                                 

        for i=1:N

            %All indices of vertices which do not belong to the same class with ith vertex
            VertexSet = find(y(1,:)~=y(1,i));                       

            %Similarities between vertex i and vertices of VertexSet
            InterClassSimilarities = S(i,VertexSet);          

            [~,ids] = sort(InterClassSimilarities,'descend');

            %kp might exceed the number of outside neighbors
            p = min(length(VertexSet),kp);

            %If q belongs to the k nearest neighbors of w outside the class we put one
            Wp(i,VertexSet(ids(1:p))) = 1;                              

        end

        %We turn the undirected penaly graph into directed
        for i=1:N

            for j=1:N

                %If i belongs to the k nearest neighbors of j or j belongs to the k nearest neighbors of i,
                %we put an edge between the two nodes
                if(Wp(i,j)==1 || Wp(j,i)==1)    

                    Wp(i,j) = 1;  

                    Wp(j,i) = 1;

                end

            end

        end        
        
        %The diagonal Degree matrix
        D  = diag(sum(W));      
        L = D - W;      
        
        %The diagonal Degree matrix
        D  = diag(sum(Wp));      
        Lp = D - Wp;          
        
        
    case 'CDA'
        
        %The number of samples
        N = size(y,2);

        %Total number of classes
        NumOfClasses = max(y(1,:));

        %List with the number of clusters belonging to each class
        NumOfClusters = zeros(1,NumOfClasses);

        %Cardinal numbers of clusters of several classes
        Cards = cell(1,NumOfClasses);

        %Initial ids of data points in several clusters of several classes
        IDS = cell(1,NumOfClasses);

        for i=1:NumOfClasses

            %The data points which belong to ith class
            ClassIds = find(y(1,:)==i);
            Class = y(2,ClassIds); 

            %The number of clusters in the ith class
            NumOfClusters(i) = max(Class);

            %For every cluster of ith class we have a cardinality
            Cards{i} = zeros(1,NumOfClusters(i));

            %For every class i, there is a list of ids of data points which belong to a specific cluster
            IDS{i} = cell(1,NumOfClusters(i));

            for j=1:NumOfClusters(i)

                %The specific cluster of specific class data points initial ids
                IDS{i}{j} = ClassIds(Class==j);

                %The corresponding cardinalities
                Cards{i}(j) = length(IDS{i}{j});

            end

        end
        
        %The total number of clusters on the whole data set
        TotalClusters = sum(NumOfClusters);

        %The intrinsic and the penalty graph matrices
        W = zeros(N,N);
        Lp = zeros(N,N);

        for i=1:NumOfClasses

            for j=1:NumOfClasses

                for p=1:NumOfClusters(i)

                    for q=1:NumOfClusters(j)

                        if(i==j && p==q)
                                                        
                            %The sum of the clusters in all classes except for i
                            Coef = TotalClusters - NumOfClusters(i);

                            Lp(IDS{i}{p},IDS{i}{p}) = Coef / ((Cards{i}(p))^2);

                            W(IDS{i}{p},IDS{i}{p}) = 1 / (Cards{i}(p));

                        elseif(i==j && p~=q)

                            %For data points of different clusters but of the same class we put zero
                            Lp(IDS{i}{p},IDS{i}{q}) = 0;

                        else

                            %This is the case of data points which belong to different classes 
                            Lp(IDS{i}{p},IDS{j}{q}) = -1 / ((Cards{i}(p)) * (Cards{j}(q)));

                        end

                     end

                 end

             end

        end   
                
        L  = eye(N,N) - W;      
        
        
    case 'SDA'
        
        %The number of samples
        N = size(y,2);

        %Total number of classes
        NumOfClasses = max(y(1,:));

        %List with the number of clusters belonging to each class
        NumOfClusters = zeros(1,NumOfClasses);

        %Cardinal numbers of clusters of several classes
        Cards = cell(1,NumOfClasses);

        %Initial ids of data points in several clusters of several classes
        IDS = cell(1,NumOfClasses);

        for i=1:NumOfClasses

            %The data points which belong to ith class
            ClassIds = find(y(1,:)==i);
            Class = y(2,ClassIds); 

            %The number of clusters in the ith class
            NumOfClusters(i) = max(Class);

            %For every cluster of ith class we have a cardinality
            Cards{i} = zeros(1,NumOfClusters(i));

            %For every class i, there is a list of ids of data points which belong to a specific cluster
            IDS{i} = cell(1,NumOfClusters(i));

            for j=1:NumOfClusters(i)

                %The specific cluster of specific class data points initial ids
                IDS{i}{j} = ClassIds(Class==j);

                %The corresponding cardinalities
                Cards{i}(j) = length(IDS{i}{j});

            end

        end

        %The penalty graph matrix
        Lp = zeros(N,N);

        for i=1:NumOfClasses

            for j=1:NumOfClasses

                for p=1:NumOfClusters(i)

                    for q=1:NumOfClusters(j)

                        if(i==j && p==q)
                                                        
                            Lp(IDS{i}{p},IDS{i}{p}) = (N - length(find(y(1,:)==i))) / (N * (Cards{i}(p)));

                        elseif(i==j && p~=q)

                            %For data points of different clusters but of the same class we put zero
                            Lp(IDS{i}{p},IDS{i}{q}) = 0;

                        else

                            %This is the case of data points which belong to different classes 
                            Lp(IDS{i}{p},IDS{j}{q}) = -1/N;

                        end

                     end

                 end

             end

        end   
        
        %The intrinsic matrix, which is essentially the covariance matrix
        %of the data
        L = eye(N,N) - (1/N) * ones(N,N);
        
end