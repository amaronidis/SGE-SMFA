function [Type,Mode,pcaDim,Par,Kernel] = SGE_AdjustParameterCV()

Type      = input('Set the type of Cross Validation procedure (LOSO, LOPO, LODO, n-fold):  ','s');
if(strcmp(Type,'n-fold')==1)
    Type  = input('Set the number of folds:  ');
end
Mode      = input('Set the mode of Cross Validation procedure (S (Straight), R (Reverse)):  ','s');

pcaDim     = input('Set desired PCA energy as preprocessing step (0 for no PCA):  ');

Par.Metric = input('Set metric for assessment(euc, mah1 or mah2):  ','s');

Kernel = input('Kernel? (0: No Kernel, >0: similarity multiplying factor)');
    
SubclassFlag = 0;

   iter = 1;
while(iter~=0)
    
method = input('Choose dimensionality reduction method (LPP, PCA, LDA, MFA, CDA, SDA, SMFA, 0:if finished):  ','s');
      
switch method
    
    case 'LPP'
        
        sigma       = input('LPP-->  Set sigma for neighborhood modeling:  ');
        dim         = input('LPP-->  Set maximum retained dimension:  ');
        k           = input('LPP-->  Set parameter k for classification:  ');
        fprintf('\n')

        Par.LPP.Sigma = sigma;
        Par.LPP.dim = dim;
        Par.LPP.k   = k;    
    
    case 'PCA'
        
        dim         = input('PCA-->  Set maximum retained dimension:  ');
        k           = input('PCA-->  Set parameter k for classification:  ');
        fprintf('\n')

        Par.PCA.dim = dim;
        Par.PCA.k   = k;

    case 'LDA'
        
        k           = input('LDA-->  Set parameter k for classification:  ');
        fprintf('\n')
        
        Par.LDA.k   = k; 
        
    case 'MFA'
        
        kInt        = input('MFA-->  Set intrinsic parameter:  ');
        kPen        = input('MFA-->  Set penalty parameter:  ');
        k           = input('MFA-->  Set parameter k for classification:  ');
        fprintf('\n')
        
        Par.MFA.kInt        = kInt;
        Par.MFA.kPen        = kPen;
        Par.MFA.k           = k;
        
    case 'SMFA'
        
        if(SubclassFlag==0)
            ClusterMethod = input('SMFA-->  Select clustering method: KM (K-Means), MSC (Multiple Spectral Clustering):  ','s');
            Par.ClusterMethod = ClusterMethod;

            if(strcmp(ClusterMethod,'MSC')==1)
                SimilCoef   = input('SMFA-->  Set Similarity coefficient:  ');
                PlausThres  = input('SMFA-->  Set Plausibility Threshold:  ');
                LCardThres  = input('SMFA-->  Set Lower Cardinality Threshold:  ');
                UCardThres  = input('SMFA-->  Set Upper Cardinality Threshold:  ');
                CardThres   = [LCardThres UCardThres]; 

                Par.SimilCoef = SimilCoef;
                Par.SMFA.PlausThres  = PlausThres;
                Par.SMFA.CardThres   = CardThres; 
            elseif(strcmp(ClusterMethod,'KM')==1)
                Par.SMFA.KM = input('SMFA-->  Set number of clusters per class in a vector form ([n1,n2,...,nc]):  ');
            end    
            SubclassFlag = 1;
        end
        
        Par.SMFA.kInt = input('SMFA-->  Set intrinsic parameter:  ');
        Par.SMFA.kPen = input('SMFA-->  Set penalty parameter:  ');
        Par.SMFA.k    = input('SMFA-->  Set parameter k for classification:  ');
        fprintf('\n')              
        
    case 'CDA'
        
        if(SubclassFlag==0)
            ClusterMethod = input('CDA-->  Select clustering method: KM (K-Means), MSC (Multiple Spectral Clustering):  ','s');
            Par.ClusterMethod = ClusterMethod;

            if(strcmp(ClusterMethod,'MSC')==1)
                SimilCoef   = input('CDA-->  Set Similarity coefficient:  ');
                PlausThres  = input('CDA-->  Set Plausibility Threshold:  ');
                LCardThres  = input('CDA-->  Set Lower Cardinality Threshold:  ');
                UCardThres  = input('CDA-->  Set Upper Cardinality Threshold:  ');
                CardThres   = [LCardThres UCardThres]; 

                Par.SimilCoef = SimilCoef;
                Par.CDA.PlausThres  = PlausThres;
                Par.CDA.CardThres   = CardThres; 
            elseif(strcmp(ClusterMethod,'KM')==1)
                Par.CDA.KM = input('CDA-->  Set number of clusters per class in a vector form ([n1,n2,...,nc]):  ');
            end  
            SubclassFlag = 1;
        end

        Par.CDA.k    = input('CDA-->  Set parameter k for classification:  ');
        fprintf('\n')         
        
    case 'SDA'
        
        if(SubclassFlag==0)
            ClusterMethod = input('SDA-->  Select clustering method: KM (K-Means), MSC (Multiple Spectral Clustering):  ','s');
            Par.ClusterMethod = ClusterMethod;

            if(strcmp(ClusterMethod,'MSC')==1)
                SimilCoef = input('SDA-->  Set Similarity coefficient:  ');
                PlausThres  = input('SDA-->  Set Plausibility Threshold:  ');
                LCardThres  = input('SDA-->  Set Lower Cardinality Threshold:  ');
                UCardThres  = input('SDA-->  Set Upper Cardinality Threshold:  ');
                CardThres   = [LCardThres UCardThres]; 

                Par.SimilCoef = SimilCoef;
                Par.SDA.PlausThres  = PlausThres;
                Par.SDA.CardThres   = CardThres; 
            elseif(strcmp(ClusterMethod,'KM')==1)
                Par.SDA.KM = input('SDA-->  Set number of clusters per class in a vector form ([n1,n2,...,nc]):  ');
            end     
            SubclassFlag = 1;
        end

        Par.SDA.k    = input('SDA-->  Set parameter k for classification:  ');
        fprintf('\n')   
        
    otherwise
        
        iter = 0;
        
end

end