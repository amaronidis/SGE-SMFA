function [X,y,XTest,yTest,Clusters] = RandomDataGenerator(Dim,d,maxC,maxS,Range,N,p)

%RANDOM

%This function constructs data from random Gaussian distributions
%
%Inputs->           Dim:        Dimensionality of data
%                   d:          Degree of difficulty (1:30)
%                   maxC:      Maximum number of classes
%                   maxS:      Maximum number of subclasses (clusters)
%                   Range:      [a b] where a, b are integers. [a b] is the range to get a random multiplier m. 
%                               m*dim adjusts the number of a specific subclass of a specific class.
%                   N:          An integer which multiplied with dim gives us the number of samples in every sublass for the test set
%                   p:          1 if you want plot, 0 if not
%
%Outputs->          X:          Training set
%                   y:          Training labels
%                   XTest:     Testing set
%                   yTest:     Testing labels
%                   Clusters:   Row vector with the numbers of clusters in every class

X = [];

y = [];

XTest = [];

yTest = [];

%Number of testing samples in every specific class
N_test = N * Dim;

%Number of classes
classes = randi([2,maxC]);

%Row vector with the number of clusters in every class
Clusters = zeros(1,classes);

for i=1:classes 
    
    %Number of subclasses (clusters) in the specific class
    subclasses = randi([1,maxS]);
    
    Clusters(i) = subclasses;
    
    %We repeat this variable to be able to change it inside the loop without changing the global one N_test
    N_test_class = N_test;
    
    for k=1:subclasses
          
        %Define a random mean for gaussian distribution
        mean = randi([-12,12],Dim,1);
        
        %in order to produce also negative covariance
        sigma = 2 * rand(Dim,Dim) - 1; 
        
        %in order to have different scales in the subclasses
        cov = (1 + d*rand) * sigma * sigma'; 
        
        %Number of training samples in specific subclass. We allow for
        %every subclass to have different number of samples
        N_sub = randi(Range) * Dim;
        
        %Gradually onstruct training set
        data = mvnrnd(mean',cov,N_sub)';
        
        %We distribute the standard number of class testing samples to the several subclasses
        if(k<subclasses && subclasses>1)
            
            N_test_sub = randi([0 N_test_class]);
            N_test_class = N_test_class - N_test_sub;
        
        else
            
            N_test_sub = N_test_class;
            
        end
        
        %We gradually construct the testing set
        test_data = mvnrnd(mean',cov,N_test_sub)';
       
        %Training labels
        labels = i*ones(1,N_sub);
        
        %Testing labels
        test_labels = i*ones(1,N_test_sub);
        
        X = [X data];
        
        XTest = [XTest test_data];
        
        y = [y labels];
        
        yTest = [yTest test_labels];
        
    end
    
end

if(p==1)
    
    figure
    plot(X(1,y==1),X(2,y==1),'b+',X(1,y==2),X(2,y==2),'r+',X(1,y==3),X(2,y==3),'g+')
    figure
    plot(XTest(1,yTest==1),XTest(2,yTest==1),'b+',XTest(1,yTest==2),XTest(2,yTest==2),'r+',XTest(1,yTest==3),XTest(2,yTest==3),'g+')

end