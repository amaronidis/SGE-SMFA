function [Clusters,Measures,DELTA,KAPPA,EigVals] = SGE_MultiSpecCluster(X,S,pt,ct)

%MULTI SPEC CLUSTER

%This function performs Algorithm 2 of paper 'Spectral Methods for Automatic Multiscale Data Clustering'
%
%Inputs->           X:  Data Matrix, (M x N, where M is the dimensionality and N is the number of samples)
%                   S:  Similarity Matrix
%                   pt: Plausibility threshold. Algorithm outputs only partitions which have plausibility measure over pt
%                   ct: Cardinality threshold multiplier. T = ct * (N/K), where T is the threshold, ct the multiplier, N the number of
%                       samples and K the number of clusters
%                       ct can have the form [T_min T_max]. In this case every cluster must have num of samples between T_min and T_max
%
%Outputs->          Clusters:   A struct with fields-->
%                                       Partitions: The final clustered Data Matrix
%                                       Labels:     Labels of each partition
%                                       InitIds:    The ids of the samples before clustering
%                   Measures:   A struct with fields-->
%                                       Cardinality:  A row vector with 0s when the corresponding partitions pass cardinality threshold
%                                                     and 1s when not
%                                       Plausibility: A row vector with the plausibility measures of the corresponding partitions
%                                       Stability:    A row vector with the stability measures of the corresponding partitions       
%                   DELTA:      The several delta for every M (see related paper)
%                   KAPPA:      The corresponding K (see related paper)
%                   EigVals:    The eigenvalues of Laplacian

N = size(X,2);

[~,EigVals] = SGE_LaplacianEigenanalysis(S);                            

[Plausible,M_max,DELTA,KAPPA] = SGE_FindPlausible(EigVals);  

%We extract the plausibility measure for the several partitions
b = Plausible.Delta;

%How many partitions
L = length(b); 

%We keep only the plausibilities for the partitions that pass the threshold test
IdsPass = find(b>=pt);
b       = b(IdsPass);

if (isempty(b)==1)
    fprintf('\nNo partitions found\n')
    
    %Saves the several partitions in a cell
    Clusters{1,1}.Partitions = X;
    
    %Saves the several labels in a cell
    Clusters{1,1}.Labels = ones(1,size(X,2));
    
    %Saves the several ids in a cell
    Clusters{1,1}.InitIds = 1:size(X,2);
    
    %Of course if no partitions found, then the cardinality threshold is
    %passed by definition
    Measures.Cardinality = 1;
    
    %Stability measure
    Measures.Stability = 1;
    
    %Plausibility measure
    Measures.Plausibility = 1;
    
    return
    
end 

%Stability measures
a = zeros(1,L);

a(1) = Plausible.M(1)/M_max;

%Here we calculate the stability measure for the several partitions
for j=2:L
    a(j) = (Plausible.M(j) - Plausible.M(j-1))/M_max;
end

%We hold only the stabilities for the partitions that pass the threshold test
a = a(IdsPass);

fprintf('\nScales of partition:    %d',L)
fprintf('\nPlausibility threshold: %.2f',pt)
fprintf('\nPass:                   %d',length(b))
fprintf('\n\nPartition    Clusters     Stability       Plausibility     CT pass\n')

%We preallocate these variables for speed
Clusters = cell(1,L);

%1 if partition passes cardinality test, 0 if not (for every partition)
Cardinality = zeros(1,length(b));

%Gives results for every partition 1:L
    for i=1:length(b)

        %Use maximin to look for the most representative centers
        InitialCenters = Maximin(X,Plausible.Kappa(IdsPass(i)));
        
        %Use K-means clustering in initial space
        IDX = kmeans(X',Plausible.Kappa(IdsPass(i)),'maxiter',10000,'start',InitialCenters');  
        
        %Cardinality threshold
        
        test = 1;
        
        if(size(ct,2)==1) 
            
            T = ct * (N/Plausible.Kappa(IdsPass(i)));
            
            %This loop checks the cardinalities of diverse clusters
            for p=1:Plausible.Kappa(IdsPass(i))
                if(length(find(IDX==p))<T)
                    test = 0;
                    break
                end
            end
            
        else
            
            T_min = ct(1);
            T_max = ct(2);
            
            %This loop checks the cardinalities of diverse clusters
            for p=1:Plausible.Kappa(IdsPass(i))
                if(length(find(IDX==p))<T_min || length(find(IDX==p))>T_max)
                    test = 0;
                    break
                end
            end
                        
        end
                      
        Cardinality(i) = test;
        
        Measures.Cardinality  = Cardinality;
        Measures.Stability    = a;
        Measures.Plausibility = b;
                
        %Get IDX in a block type vector
        [IDXBlock,ids] = sort(IDX);   
        
        %Built the final Data Matrix in a block-form
        C = X(:,ids);                    
        
        %y contains the corresponding labels of points of C
        y = IDXBlock';                 
        
        ids = ids';
                                                                                                                                                                                                                                                                                       
        Clusters{1,i}.Partitions = C;
    
        Clusters{1,i}.Labels     = y;
    
        Clusters{1,i}.InitIds    = ids;
    
        fprintf('   %2d           %2d           %2.2f             %2.2f            %d\n',i,Plausible.Kappa(IdsPass(i)),a(i),b(i),test)

    end

end