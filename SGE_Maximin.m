function InitialCenters = SGE_Maximin(X,K)

%MAXIMIN

%This function performs the maximin algorithm for finding plausible initial
%centers before applying the k-means algorithm
%
%Inputs->           X: Data Matrix (M x N, where M is the dimensionality and N is the number of samples)
%                   K: Number of initial centers and generally the total number of clusters
%                   
%Output->           InitialCenters: A (M x K) matrix with the initial
%centers as columns

%M: Dimensionality  N: Number of samples
[M,N] = size(X);                                        

DistMatrix = pdist2(X',X');  

%We randomly choose an integer between 1 and N
r = randi(N);                                        

%This matrix will be populated by the initial centers
InitialCenters = zeros(M,K);                           
    
%This list indicates which vectors have been used for initial centers (1 if used, 0 if not used)
CheckList = zeros(1,N);                            


%First step

%We arbitrarily choose the first center
InitialCenters(:,1) = X(:,r);                       

%If one vector consists an initial center, the corresponding place in the CheckList list becomes 1
CheckList(r) = 1;                                


%Second step

%This list contains the indices of the unused for the moment vectors
UnusedVecs = find(CheckList==0);                

%The distances between every unused vector and the used one
TestDist = DistMatrix(r,UnusedVecs);  

%We find the vector which lies furthest apart from the used one
[~,MaxId] = max(TestDist);                

%This vector consists the second initial center
InitialCenters(:,2) = X(:,UnusedVecs(MaxId));     

%Of course the corresponding place of the checking_list becomes 1
CheckList(UnusedVecs(MaxId)) = 1;              


%Third step

step = 3;

while(step<K+1)

%This list contains the indices of the used for the moment vectors
UsedVecs = find(CheckList==1);                  

%This list contains the indices of the unused for the moment vectors
UnusedVecs = find(CheckList==0);                

%Like above
TestDist = DistMatrix(UsedVecs,UnusedVecs);  

%Now, for every unused vector, we first search the min distance between that and the used ones 
[MinDist,~] = min(TestDist);               

%Then we try to find the max of those min distances
[~,MaxId] = max(MinDist);                      

%And we mount the initial_centers matrix with the vector which realizes the above max of min distances
InitialCenters(:,step) = X(:,UnusedVecs(MaxId));  

CheckList(UnusedVecs(MaxId)) = 1;

step = step + 1;

end
