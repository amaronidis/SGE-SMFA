function CrossValid = SGE_CrossValidation(X,y,Type,pcaDim,Par,Mode,Kernel)

%CROSS VALIDATION

%This function conducts cross validation on a given set of patterns.
%
%Inputs->               X:      The Data Matrix (M x N, where M is the dimensionality and N is the number of samples)
%                       y:      A row vector with the labels of the samples (1 x N)
%                       Type:   An integer n, for n-Fold Cross Validation
%                               'LOPO', for Leave One Person Out Cross Validation*
%                               'LODO', for Leave One Day (Session) Out Cross Validation*
%                               'LOSO', for Leave One Sample Out Cross Validation
%                                           * In order to perform the above types of
%                                           cross validation the variable y should be
%                                           (3 x N)->
%                                                       1st row: class
%                                                       2nd row: person (or subject)
%                                                       3rd row: day (or session)
%                       pcaDim: Optional PCA step-> Integer for number of dimensions.
%                                                   Real in interval (0,1) for the energy(%) kept
%                                                   0 for no PCA
%                       Par:    All needed parameters in a multi-struct (use SGE_AdjustParameterCV.m)
%                       Kernel: 0 for linear case, 1 for kernel case
%
%                       Mode: 'Straight' uses the small part of the data as testing set
%                             'Reverse' uses the big part of the data as test set
%
%Outputs->              CrossValid

MethodsOn = fieldnames(Par);

fprintf('Cross Validation\n')

NumOfClasses = max(y(1,:));
InitDim = size(X,1);

%The maximum common available dimension for every cv step
LPPMaxCommonDim = inf;
CDAMaxCommonDim = inf;
PCAMaxCommonDim = inf;
LDAMaxCommonDim = inf;
SDAMaxCommonDim = inf;
MFAMaxCommonDim = inf;
SMFAMaxCommonDim = inf;
%The maximum available dimension for some cv step
CDAMaxDim = 0;
SDAMaxDim = 0;
SMFAMaxDim = 0;

if(strcmp(Type,'LOPO')==1)
    nFold = max(y(2,:));
elseif(strcmp(Type,'LODO')==1)
    nFold = max(y(3,:));
elseif(strcmp(Type,'LOSO')==1)
    nFold = size(y,2);
else
    nFold = Type;
end

for i=1:nFold
    
    if(strcmp(Type,'LOPO')==1)
        TrainIds = find(y(2,:)~=i);
        TestIds = find(y(2,:)==i);
    elseif(strcmp(Type,'LODO')==1)
        TrainIds = find(y(3,:)~=i);
        TestIds = find(y(3,:)==i);
    elseif(strcmp(Type,'LOSO')==1)
        TrainIds = 1:size(y,2);
        TrainIds(i) = [];
        TestIds = i;
    else
        [TrainIds,TestIds] = SGE_DivisionIds(y,nFold,i);
    end
       
    if(strcmp(Mode,'R')==1)
    
    %%%%%%%%%%%%%%%%%%%
    temp = TrainIds;
    TrainIds = TestIds;
    TestIds = temp;
    %%%%%%%%%%%%%%%%%%%
    
    end
    
    CrossValid.Folds{i}.DataInfo.TrainIds = TrainIds;
    CrossValid.Folds{i}.DataInfo.TestIds = TestIds;    
    
    fprintf('\nStep: %2d /%2d --> ',i,nFold)
    
    %First divide the data set into the trainig and the testing sets
    [XTrain,XTest] = SGE_CrossDivision(X,'data',TrainIds,TestIds);
    [yTrain,yTest] = SGE_CrossDivision(y,'data',TrainIds,TestIds);
    
    %The extra information about the persons and the days will not be used further
    yTrain = yTrain(1,:);
    yTest  = yTest(1,:);
    
    clear TrainIds;
    clear TestIds;
    
    InitDim = size(XTrain,1);
    
    CrossValid.Folds{i}.DataInfo.InitDim = InitDim;  
    
    
    
    if(Kernel>0)
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %KERNELIZATION
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            D = pdist2(XTrain',XTrain');
            variance = Kernel * mean(D(:));
            KTrain = SGE_GramMatrix(XTrain,'gau',variance);
            XTest  = SGE_GramMatrix(XTrain,XTest,'gau',variance);
            XTrain = KTrain;
            clear KTrain;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            
    end    
    
    
    %Optional PCA preprocessing step
    if(pcaDim>0)
        
        [XTrain,PC,PV] = pca(XTrain,pcaDim);
        XTest = PC' * XTest;
        newDim = size(XTrain,1);
        
        CrossValid.Folds{i}.pcaPreprocess.newDim = newDim;
        CrossValid.Folds{i}.pcaPreprocess.MappedTrain = XTrain;
        CrossValid.Folds{i}.pcaPreprocess.MappedTest = XTest;
        CrossValid.Folds{i}.pcaPreprocess.TransMatrix = PC;
        CrossValid.Folds{i}.pcaPreprocess.EigVals = PV;
        
    elseif(pcaDim==0)
        
        newDim = InitDim;
        
    end        
    
    
    %LPP
    I = find(strcmp(MethodsOn,'LPP')==1);
    if(isempty(I)==0)
        
        fprintf('\nRunning LPP');
        
        %Parameter Collection
        par.mode  = 'LPP';
        par.X     = XTrain;
        par.sigma = Par.LPP.Sigma;
        dim = Par.LPP.dim;
        if(dim>newDim)
            dim = newDim - 1;
        end
        
        LPPMaxCommonDim = min(LPPMaxCommonDim,dim);
        MaxTrainRate    = 0;
        MaxTestRate     = 0;
        
        k = Par.LPP.k;
                       
        [W,Wp] = SGE_GraphConstruct(yTrain,par);

        [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);
        
        clear W;

        for j=1:dim
            
            MappedX = Projection(XTrain,1:j,TransMatrix);
            MappedU = Projection(XTest,1:j,TransMatrix);
                                                         
            [TrainRates,TestRates] = SGE_Assessment(MappedX,yTrain,MappedU,yTest,k,Par.Metric);
            
            if(TrainRates.TotalRate>MaxTrainRate)
                MaxTrainRate    = TrainRates.TotalRate;
                MaxTrainRateDim = j;
            end
            
            if(TestRates.TotalRate>MaxTestRate)
                MaxTestRate    = TestRates.TotalRate;
                MaxTestRateDim = j;
            end
            
            CrossValid.Folds{i}.DRMethods.LPP.Mapped{j}.TrainSet = MappedX;
            CrossValid.Folds{i}.DRMethods.LPP.Mapped{j}.TestSet  = MappedU;
            
            clear MappedX;
            clear MappedU;            
            
            CrossValid.Folds{i}.DRMethods.LPP.TrainRates{j} = TrainRates;
            CrossValid.Folds{i}.DRMethods.LPP.TestRates{j}  = TestRates;
            
            clear TrainRates;
            clear TestRates;
        
        end
        
        CrossValid.Folds{i}.DRMethods.LPP.TransMatrix     = TransMatrix(:,1:dim);
        CrossValid.Folds{i}.DRMethods.LPP.EigVals         = EigVals(1:dim);
        
        CrossValid.Folds{i}.DRMethods.LPP.MaxTrainRate    = MaxTrainRate;
        CrossValid.Folds{i}.DRMethods.LPP.MaxTrainRateDim = MaxTrainRateDim;
        
        CrossValid.Folds{i}.DRMethods.LPP.MaxTestRate     = MaxTestRate;
        CrossValid.Folds{i}.DRMethods.LPP.MaxTestRateDim  = MaxTestRateDim;
        
        clear TransMatrix;
        clear EigVals;
        
    end
    
    
    %PCA
    I = find(strcmp(MethodsOn,'PCA')==1);
    if(isempty(I)==0)
        
        fprintf('\nRunning PCA');
        
        %Parameter Collection
        par.mode = 'PCA';
        dim = Par.PCA.dim;
        if(dim>InitDim)
            dim = InitDim - 1;
        end
        
        PCAMaxCommonDim = min(PCAMaxCommonDim,dim);
        MaxTrainRate    = 0;
        MaxTestRate     = 0;
        
        k = Par.PCA.k;
        
        [~,TransMatrix,EigVals] = pca(XTrain,dim);
                       
%         [W Wp] = GraphConstruct(yTrain,par);
% 
%         [TransMatrix EigVals] = Mapping(XTrain,dim,W,Wp);
%         
%         clear W;

        for j=1:dim
            
            MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
            MappedU = SGE_Projection(XTest,1:j,TransMatrix);
                                                         
            [TrainRates,TestRates] = SGE_Assessment(MappedX,yTrain,MappedU,yTest,k,Par.Metric);
            
            if(TrainRates.TotalRate>MaxTrainRate)
                MaxTrainRate    = TrainRates.TotalRate;
                MaxTrainRateDim = j;
            end
            
            if(TestRates.TotalRate>MaxTestRate)
                MaxTestRate    = TestRates.TotalRate;
                MaxTestRateDim = j;
            end
            
            CrossValid.Folds{i}.DRMethods.PCA.Mapped{j}.TrainSet = MappedX;
            CrossValid.Folds{i}.DRMethods.PCA.Mapped{j}.TestSet  = MappedU;
            
            clear MappedX;
            clear MappedU;            
            
            CrossValid.Folds{i}.DRMethods.PCA.TrainRates{j} = TrainRates;
            CrossValid.Folds{i}.DRMethods.PCA.TestRates{j}  = TestRates;
            
            clear TrainRates;
            clear TestRates;
        
        end
        
        CrossValid.Folds{i}.DRMethods.PCA.TransMatrix     = TransMatrix(:,1:dim);
        CrossValid.Folds{i}.DRMethods.PCA.EigVals         = EigVals(1:dim);
        
        CrossValid.Folds{i}.DRMethods.PCA.MaxTrainRate    = MaxTrainRate;
        CrossValid.Folds{i}.DRMethods.PCA.MaxTrainRateDim = MaxTrainRateDim;
        
        CrossValid.Folds{i}.DRMethods.PCA.MaxTestRate     = MaxTestRate;
        CrossValid.Folds{i}.DRMethods.PCA.MaxTestRateDim  = MaxTestRateDim;
        
        clear TransMatrix;
        clear EigVals;
        
    end
    
    
    
%     if(Kernel>0)
%         
%         %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %     %KERNELIZATION
%         %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %     D = pdist2(XTrain',XTrain');
%         %     variance = Kernel * max(D(:));
%         %     KTrain = GramMatrix(XTrain,'gau',variance);
%         %     KTest  = GramMatrix(XTrain,XTest,'gau',variance);
%         %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %KERNELIZATION
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             D = pdist2(XTrain',XTrain');
%             STrain = GramMatrix(XTrain,'gau',0.12*max(D(:)));
%             %STrain = CrossDivision(S,'gram',TrainIds,TrainIds);
%             ClustersOfClasses = ClusterExtract(XTrain,yTrain,STrain,0,[5 10000]);
%             yy = CDALabelsConstruct(ClustersOfClasses,yTrain);
%             [XTrain C] = GramMatrixCDATrain(XTrain,yy,Kernel);
%             XTest  = GramMatrixCDATest(XTrain,yy,XTest,C);
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             
%     end
    
      
    
    %LDA
    I = find(strcmp(MethodsOn,'LDA')==1);
    if(isempty(I)==0)
        
        fprintf('\nRunning LDA');
        
        %Parameter Collection
        par.mode = 'LDA';
        dim = NumOfClasses - 1;
        if(dim>newDim)
            dim = newDim - 1;
        end
        
        LDAMaxCommonDim = min(LDAMaxCommonDim,dim);
        MaxTrainRate    = 0;
        MaxTestRate     = 0;
        
        k = Par.LDA.k;
        
        [W,Wp] = SGE_GraphConstruct(yTrain,par);
                
        [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);
        
        clear W;
        clear Wp;
        
        for j=1:dim
            
            MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
            MappedU = SGE_Projection(XTest,1:j,TransMatrix);
            
            [TrainRates,TestRates] = SGE_Assessment(MappedX,yTrain,MappedU,yTest,k,Par.Metric);

            if(TrainRates.TotalRate>MaxTrainRate)
                MaxTrainRate    = TrainRates.TotalRate;
                MaxTrainRateDim = j;
            end
            
            if(TestRates.TotalRate>MaxTestRate)
                MaxTestRate    = TestRates.TotalRate;
                MaxTestRateDim = j;
            end
            
            CrossValid.Folds{i}.DRMethods.LDA.Mapped{j}.TrainSet = MappedX;
            CrossValid.Folds{i}.DRMethods.LDA.Mapped{j}.TestSet  = MappedU;                          
                      
            clear MappedX;
            clear MappedU;
             
            CrossValid.Folds{i}.DRMethods.LDA.TrainRates{j} = TrainRates;
            CrossValid.Folds{i}.DRMethods.LDA.TestRates{j} = TestRates;
            
            clear TrainRates;
            clear TestRates;
            
        end
        
        CrossValid.Folds{i}.DRMethods.LDA.TransMatrix     = TransMatrix(:,1:dim);
        CrossValid.Folds{i}.DRMethods.LDA.EigVals         = EigVals(1:dim);
        
        CrossValid.Folds{i}.DRMethods.LDA.MaxTrainRate    = MaxTrainRate;
        CrossValid.Folds{i}.DRMethods.LDA.MaxTrainRateDim = MaxTrainRateDim;
        
        CrossValid.Folds{i}.DRMethods.LDA.MaxTestRate     = MaxTestRate;
        CrossValid.Folds{i}.DRMethods.LDA.MaxTestRateDim  = MaxTestRateDim;

        clear TransMatrix;
        clear EigVals;
        
    end
    
    
    %MFA
    I = find(strcmp(MethodsOn,'MFA')==1);
    if(isempty(I)==0)
        
        fprintf('\nRunning MFA');
        
        %Parameter Collection
        par.mode = 'MFA';
        par.SimilMatrix = -pdist2(XTrain',XTrain');
        par.kInt = Par.MFA.kInt;
        par.kPen = Par.MFA.kPen;
        dim = Par.MFA.kPen;
        if(dim>newDim)
            dim = newDim - 1;
        end
        
        MFAMaxCommonDim = min(MFAMaxCommonDim,dim);
        MaxTrainRate    = 0;
        MaxTestRate     = 0;
        
        k = Par.MFA.k;
        
        [W,Wp] = SGE_GraphConstruct(yTrain,par);
        
        [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);
        
        clear W;
        clear Wp;
        
        for j=1:dim
            
            MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
            MappedU = SGE_Projection(XTest,1:j,TransMatrix);
                    
            [TrainRates, TestRates] = SGE_Assessment(MappedX,yTrain,MappedU,yTest,k,Par.Metric);
            
            if(TrainRates.TotalRate>MaxTrainRate)
                MaxTrainRate    = TrainRates.TotalRate;
                MaxTrainRateDim = j;
            end
            
            if(TestRates.TotalRate>MaxTestRate)
                MaxTestRate    = TestRates.TotalRate;
                MaxTestRateDim = j;
            end
            
            CrossValid.Folds{i}.DRMethods.MFA.Mapped{j}.TrainSet = MappedX;
            CrossValid.Folds{i}.DRMethods.MFA.Mapped{j}.TestSet  = MappedU;              
            
            clear MappedX;
            clear MappedU;
            
            CrossValid.Folds{i}.DRMethods.MFA.TrainRates{j} = TrainRates;
            CrossValid.Folds{i}.DRMethods.MFA.TestRates{j} = TestRates;            
            
            clear TrainRates;
            clear TestRates;
            
        end
        
        CrossValid.Folds{i}.DRMethods.MFA.TransMatrix     = TransMatrix(:,1:dim);
        CrossValid.Folds{i}.DRMethods.MFA.EigVals         = EigVals(1:dim);        
        
        CrossValid.Folds{i}.DRMethods.MFA.MaxTrainRate    = MaxTrainRate;
        CrossValid.Folds{i}.DRMethods.MFA.MaxTrainRateDim = MaxTrainRateDim;
        
        CrossValid.Folds{i}.DRMethods.MFA.MaxTestRate     = MaxTestRate;
        CrossValid.Folds{i}.DRMethods.MFA.MaxTestRateDim  = MaxTestRateDim;
        
        clear TransMatrix;
        clear EigVals;
        
    end
    
    
    %SUBCLASS METHODS
    I = find(strcmp(MethodsOn,'CDA')==1 | strcmp(MethodsOn,'SDA')==1 | strcmp(MethodsOn,'SGE')==1 | strcmp(MethodsOn,'SMFA')==1 | strcmp(MethodsOn,'SDMFA')==1);
    if(isempty(I)==0)
    
    
        %Subclass extraction
        if(find(strcmp(MethodsOn,'CDA')==1))
            if(strcmp(Par.ClusterMethod,'MSC')==1)
                S = Par.SimilCoef;
                pt = Par.CDA.PlausThres;   
                ct = Par.CDA.CardThres;
                D = pdist2(XTrain',XTrain');
                STrain = SGE_GramMatrix(XTrain,'gau',S*mean(D(:)));
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,STrain,pt,ct);
            elseif(strcmp(Par.ClusterMethod,'KM')==1)
                KM = Par.CDA.KM;
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,KM,0,0);
            end
        elseif(find(strcmp(MethodsOn,'SDA')==1))
            if(strcmp(Par.ClusterMethod,'MSC')==1)
                S = Par.SimilCoef;
                pt = Par.SDA.PlausThres;   
                ct = Par.SDA.CardThres;
                D = pdist2(XTrain',XTrain');
                STrain = SGE_GramMatrix(XTrain,'gau',S*mean(D(:)));
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,STrain,pt,ct);
            elseif(strcmp(Par.ClusterMethod,'KM')==1)
                KM = Par.SDA.KM;
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,KM,0,0);
            end
        elseif(find(strcmp(MethodsOn,'SMFA')==1))
            if(strcmp(Par.ClusterMethod,'MSC')==1)
                S = Par.SimilCoef;
                pt = Par.SMFA.PlausThres;   
                ct = Par.SMFA.CardThres;
                D = pdist2(XTrain',XTrain');
                STrain = SGE_GramMatrix(XTrain,'gau',S*mean(D(:)));
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,STrain,pt,ct);
            elseif(strcmp(Par.ClusterMethod,'KM')==1)
                KM = Par.SMFA.KM;
                ClustersOfClasses = SGE_SubclassExtract(XTrain,yTrain,KM,0,0);
            end
        end
     
        [yyTrain,ClustersPerClass,TotalClusters] = SGE_SubclassLabels(ClustersOfClasses,yTrain);

        CrossValid.Folds{i}.ClusterInfo.TotalClusters    = TotalClusters; 
        CrossValid.Folds{i}.ClusterInfo.ClustersPerClass = ClustersPerClass;
        CrossValid.Folds{i}.ClusterInfo.Clusters         = ClustersOfClasses;  
        
        clear ClustersOfClasses;
        
        dim = TotalClusters - 1;
        if(dim>newDim)
            dim = newDim -1;
        end

            
        %CDA
        I = find(strcmp(MethodsOn,'CDA')==1);
        if(isempty(I)==0)   
            
            fprintf('\nRunning CDA');
            
            par.mode = 'CDA';
            
            k = Par.CDA.k;

            CDAMaxCommonDim = min(CDAMaxCommonDim,dim);
            CDAMaxDim = max(CDAMaxDim,dim);
            MaxTrainRate    = 0;
            MaxTestRate     = 0;

            [W,Wp] = SGE_GraphConstruct(yyTrain,par);

            [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);

            clear W;
            clear Wp;

            for j=1:dim

                MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
                MappedU = SGE_Projection(XTest,1:j,TransMatrix);
                
                [TrainRates,TestRates] = SGE_Assessment(MappedX,yyTrain,MappedU,yTest,k,Par.Metric);
                
                if(TrainRates.TotalRate>MaxTrainRate)
                    MaxTrainRate    = TrainRates.TotalRate;
                    MaxTrainRateDim = j;
                end
            
                if(TestRates.TotalRate>MaxTestRate)
                    MaxTestRate    = TestRates.TotalRate;
                    MaxTestRateDim = j;
                end

                CrossValid.Folds{i}.DRMethods.CDA.Mapped{j}.TrainSet = MappedX;
                CrossValid.Folds{i}.DRMethods.CDA.Mapped{j}.TestSet  = MappedU;

                clear MappedX;
                clear MappedU;

                CrossValid.Folds{i}.DRMethods.CDA.TrainRates{j} = TrainRates;
                CrossValid.Folds{i}.DRMethods.CDA.TestRates{j}  = TestRates;

                clear TrainRates;
                clear TestRates;

            end
            
            Mapped.TransMatrix = TransMatrix(:,1:dim);
            Mapped.EigVals = EigVals(1:dim); 
            
            CrossValid.Folds{i}.DRMethods.CDA.MaxTrainRate    = MaxTrainRate;
            CrossValid.Folds{i}.DRMethods.CDA.MaxTrainRateDim = MaxTrainRateDim;

            CrossValid.Folds{i}.DRMethods.CDA.MaxTestRate     = MaxTestRate;
            CrossValid.Folds{i}.DRMethods.CDA.MaxTestRateDim  = MaxTestRateDim;

            clear TransMatrix;
            clear EigVals;

        end
        
        
        %SDA
        I = find(strcmp(MethodsOn,'SDA')==1);
        if(isempty(I)==0)    
            
            fprintf('\nRunning SDA');
            
            par.mode = 'SDA';
            
            k = Par.SDA.k;

            SDAMaxCommonDim = min(SDAMaxCommonDim,dim);
            SDAMaxDim = max(SDAMaxDim,dim);
            MaxTrainRate    = 0;
            MaxTestRate     = 0;

            [W,Wp] = SGE_GraphConstruct(yyTrain,par);

            [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);

            clear W;
            clear Wp;

            for j=1:dim

                MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
                MappedU = SGE_Projection(XTest,1:j,TransMatrix);
                
                [TrainRates,TestRates] = SGE_Assessment(MappedX,yyTrain,MappedU,yTest,k,Par.Metric);
                
                if(TrainRates.TotalRate>MaxTrainRate)
                    MaxTrainRate    = TrainRates.TotalRate;
                    MaxTrainRateDim = j;
                end
            
                if(TestRates.TotalRate>MaxTestRate)
                    MaxTestRate    = TestRates.TotalRate;
                    MaxTestRateDim = j;
                end

                CrossValid.Folds{i}.DRMethods.SDA.Mapped{j}.TrainSet = MappedX;
                CrossValid.Folds{i}.DRMethods.SDA.Mapped{j}.TestSet  = MappedU;

                clear MappedX;
                clear MappedU;

                CrossValid.Folds{i}.DRMethods.SDA.TrainRates{j} = TrainRates;
                CrossValid.Folds{i}.DRMethods.SDA.TestRates{j}  = TestRates;

                clear TrainRates;
                clear TestRates;

            end
            
            Mapped.TransMatrix = TransMatrix(:,1:dim);
            Mapped.EigVals = EigVals(1:dim);  
            
            CrossValid.Folds{i}.DRMethods.SDA.MaxTrainRate    = MaxTrainRate;
            CrossValid.Folds{i}.DRMethods.SDA.MaxTrainRateDim = MaxTrainRateDim;

            CrossValid.Folds{i}.DRMethods.SDA.MaxTestRate     = MaxTestRate;
            CrossValid.Folds{i}.DRMethods.SDA.MaxTestRateDim  = MaxTestRateDim;

            clear TransMatrix;
            clear EigVals;

        end
        
        
        %SMFA
        I = find(strcmp(MethodsOn,'SMFA')==1);
        if(isempty(I)==0)   
            
            fprintf('\nRunning SMFA');                 
            
            par.mode = 'SMFA';
            par.SimilMatrix = -pdist2(XTrain',XTrain');
            par.kInt = Par.SMFA.kInt;
            par.kPen = Par.SMFA.kPen;
            
            k = Par.SMFA.k;
            
         dim = Par.SMFA.kPen;
        if(dim>newDim)
            dim = newDim - 1;
        end                

            SMFAMaxCommonDim = min(SMFAMaxCommonDim,dim);
            SMFAMaxDim = max(SMFAMaxDim,dim);
            MaxTrainRate    = 0;
            MaxTestRate     = 0;

            [W,Wp] = SGE_GraphConstruct(yyTrain,par);

            [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);

            clear W;
            clear Wp;

            for j=1:dim

                MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
                MappedU = SGE_Projection(XTest,1:j,TransMatrix);
                
                [TrainRates,TestRates] = SGE_Assessment(MappedX,yyTrain,MappedU,yTest,k,Par.Metric);
                
                if(TrainRates.TotalRate>MaxTrainRate)
                    MaxTrainRate    = TrainRates.TotalRate;
                    MaxTrainRateDim = j;
                end
            
                if(TestRates.TotalRate>MaxTestRate)
                    MaxTestRate    = TestRates.TotalRate;
                    MaxTestRateDim = j;
                end

                CrossValid.Folds{i}.DRMethods.SMFA.Mapped{j}.TrainSet = MappedX;
                CrossValid.Folds{i}.DRMethods.SMFA.Mapped{j}.TestSet  = MappedU;

                clear MappedX;
                clear MappedU;

                CrossValid.Folds{i}.DRMethods.SMFA.TrainRates{j} = TrainRates;
                CrossValid.Folds{i}.DRMethods.SMFA.TestRates{j}  = TestRates;

                clear TrainRates;
                clear TestRates;

            end
            
            Mapped.TransMatrix = TransMatrix(:,1:dim);
            Mapped.EigVals = EigVals(1:dim); 
            
            CrossValid.Folds{i}.DRMethods.SMFA.MaxTrainRate    = MaxTrainRate;
            CrossValid.Folds{i}.DRMethods.SMFA.MaxTrainRateDim = MaxTrainRateDim;

            CrossValid.Folds{i}.DRMethods.SMFA.MaxTestRate     = MaxTestRate;
            CrossValid.Folds{i}.DRMethods.SMFA.MaxTestRateDim  = MaxTestRateDim;

            clear TransMatrix;
            clear EigVals;

        end       
        
        
%         %SDMFA
%         I = find(strcmp(MethodsOn,'SDMFA')==1);
%         if(isempty(I)==0)    
%             
%             par.mode = 'SDMFA';
%             par.SimilMatrix = STrain;
%             par.kInt = Par.SDMFA.kInt;
%             par.kPen = Par.SDMFA.kPen;
%             
%             k = Par.SDMFA.k;
% 
%             SDMFAMaxCommonDim = min(SDMFAMaxCommonDim,dim);
%             SDMFAMaxDim = max(SDMFAMaxDim,dim);
%             MaxTrainRate    = 0;
%             MaxTestRate     = 0;
% 
%             [W,Wp] = SGE_GraphConstruct(yyTrain,par);
% 
%             [TransMatrix,EigVals] = SGE_Mapping(XTrain,dim,W,Wp);
% 
%             clear W;
%             clear Wp;
% 
%             for j=1:dim
% 
%                 MappedX = SGE_Projection(XTrain,1:j,TransMatrix);
%                 MappedU = SGE_Projection(XTest,1:j,TransMatrix);
%                 
%                 [TrainRates,TestRates] = SGE_Assessment(MappedX,yyTrain,MappedU,yTest,k,Par.Metric);
%                 
%                 if(TrainRates.TotalRate>MaxTrainRate)
%                     MaxTrainRate    = TrainRates.TotalRate;
%                     MaxTrainRateDim = j;
%                 end
%             
%                 if(TestRates.TotalRate>MaxTestRate)
%                     MaxTestRate    = TestRates.TotalRate;
%                     MaxTestRateDim = j;
%                 end
% 
%                 CrossValid.Folds{i}.DRMethods.SDMFA.Mapped{j}.TrainSet = MappedX;
%                 CrossValid.Folds{i}.DRMethods.SDMFA.Mapped{j}.TestSet  = MappedU;
% 
%                 clear MappedX;
%                 clear MappedU;
% 
%                 CrossValid.Folds{i}.DRMethods.SDMFA.TrainRates{j} = TrainRates;
%                 CrossValid.Folds{i}.DRMethods.SDMFA.TestRates{j}  = TestRates;
% 
%                 clear TrainRates;
%                 clear TestRates;
% 
%             end
%             
%             Mapped.TransMatrix = TransMatrix(:,1:dim);
%             Mapped.EigVals = EigVals(1:dim); 
%             
%             CrossValid.Folds{i}.DRMethods.SDMFA.MaxTrainRate    = MaxTrainRate;
%             CrossValid.Folds{i}.DRMethods.SDMFA.MaxTrainRateDim = MaxTrainRateDim;
% 
%             CrossValid.Folds{i}.DRMethods.SDMFA.MaxTestRate     = MaxTestRate;
%             CrossValid.Folds{i}.DRMethods.SDMFA.MaxTestRateDim  = MaxTestRateDim;
% 
%             clear TransMatrix;
%             clear EigVals;
% 
%         end               
%         
%         
%         %SGE
%         I = find(strcmp(MethodsOn,'SGE')==1);
%         if(isempty(I)==0)    
%             
%             par.mode = 'SGE';
%             par.P    = GramMatrix(XTrain,'gau',Par.SGE.P * mean(D(:)));
%             par.Q    = ones(size(XTrain,2),size(XTrain,2));
%             
%             k = Par.SGE.k;
% 
%             SGEMaxCommonDim = min(SGEMaxCommonDim,dim);
%             SGEMaxDim = max(SGEMaxDim,dim);
%             MaxTrainRate    = 0;
%             MaxTestRate     = 0;
% 
%             [W Wp] = GraphConstruct(yyTrain,par);
% 
%             [TransMatrix EigVals] = Mapping(XTrain,dim,W,Wp);
% 
%             clear W;
%             clear Wp;
% 
%             for j=1:dim
% 
%                 MappedX = Projection(XTrain,1:j,TransMatrix);
%                 MappedU = Projection(XTest,1:j,TransMatrix);
%                 
%                 [TrainRates TestRates] = Assessment(MappedX,yyTrain,MappedU,yTest,k,Par.Metric);
%                 
%                 if(TrainRates.TotalRate>MaxTrainRate)
%                     MaxTrainRate    = TrainRates.TotalRate;
%                     MaxTrainRateDim = j;
%                 end
%             
%                 if(TestRates.TotalRate>MaxTestRate)
%                     MaxTestRate    = TestRates.TotalRate;
%                     MaxTestRateDim = j;
%                 end
% 
%                 CrossValid.Folds{i}.DRMethods.SGE.Mapped{j}.TrainSet = MappedX;
%                 CrossValid.Folds{i}.DRMethods.SGE.Mapped{j}.TestSet  = MappedU;
% 
%                 clear MappedX;
%                 clear MappedU;
% 
%                 CrossValid.Folds{i}.DRMethods.SGE.TrainRates{j} = TrainRates;
%                 CrossValid.Folds{i}.DRMethods.SGE.TestRates{j}  = TestRates;
% 
%                 clear TrainRates;
%                 clear TestRates;
% 
%             end
%             
%             Mapped.TransMatrix = TransMatrix(:,1:dim);
%             Mapped.EigVals = EigVals(1:dim); 
%             
%             CrossValid.Folds{i}.DRMethods.SGE.MaxTrainRate    = MaxTrainRate;
%             CrossValid.Folds{i}.DRMethods.SGE.MaxTrainRateDim = MaxTrainRateDim;
% 
%             CrossValid.Folds{i}.DRMethods.SGE.MaxTestRate     = MaxTestRate;
%             CrossValid.Folds{i}.DRMethods.SGE.MaxTestRateDim  = MaxTestRateDim;
% 
%             clear TransMatrix;
%             clear EigVals;
% 
%         end
        
        
        fprintf('\nComplete\n')

        clear XTrain;
        clear XTest;
        clear yTrain;
        clear yTest

    end
    
    
end
    %End of Cross Validation
    



%TOTAL RESULTS

%LPP
I = find(strcmp(MethodsOn,'LPP')==1);
if(isempty(I)==0)
    
    TrRates = zeros(LPPMaxCommonDim,nFold);
    TsRates = zeros(LPPMaxCommonDim,nFold);
    for j=1:LPPMaxCommonDim
        for i=1:nFold
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.LPP.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.LPP.TestRates{j}.TotalRate;
        end

    CrossValid.Total.LPP{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.LPP{j}.TrainRateVar = var(TrRates(j,:));
    
    CrossValid.Total.LPP{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.LPP{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%PCA
I = find(strcmp(MethodsOn,'PCA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(PCAMaxCommonDim,nFold);
    TsRates = zeros(PCAMaxCommonDim,nFold);
    for j=1:PCAMaxCommonDim
        for i=1:nFold
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.PCA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.PCA.TestRates{j}.TotalRate;
        end

    CrossValid.Total.PCA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.PCA{j}.TrainRateVar = var(TrRates(j,:));
    
    CrossValid.Total.PCA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.PCA{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%LDA
I = find(strcmp(MethodsOn,'LDA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(LDAMaxCommonDim,nFold);
    TsRates = zeros(LDAMaxCommonDim,nFold);
    for j=1:LDAMaxCommonDim
        for i=1:nFold
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.LDA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.LDA.TestRates{j}.TotalRate;
        end

    CrossValid.Total.LDA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.LDA{j}.TrainRateVar = var(TrRates(j,:));
    
    CrossValid.Total.LDA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.LDA{j}.TestRateVar = var(TsRates(j,:));

    end

end

%MFA
I = find(strcmp(MethodsOn,'MFA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(MFAMaxCommonDim,nFold);
    TsRates = zeros(MFAMaxCommonDim,nFold);
    for j=1:MFAMaxCommonDim
        for i=1:nFold
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.MFA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.MFA.TestRates{j}.TotalRate;
        end

    CrossValid.Total.MFA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.MFA{j}.TrainRateVar = var(TrRates(j,:));
    
    CrossValid.Total.MFA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.MFA{j}.TestRateVar = var(TsRates(j,:));

    end

end

%CDA
I = find(strcmp(MethodsOn,'CDA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(CDAMaxCommonDim,nFold);    
    TsRates = zeros(CDAMaxCommonDim,nFold);   
    for j=1:CDAMaxCommonDim
        for i=1:nFold
            
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.CDA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.CDA.TestRates{j}.TotalRate;
          
        end

    CrossValid.Total.CDA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.CDA{j}.TrainRateVar = var(TrRates(j,:));

    CrossValid.Total.CDA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.CDA{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%SMFA
I = find(strcmp(MethodsOn,'SMFA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(SMFAMaxCommonDim,nFold);    
    TsRates = zeros(SMFAMaxCommonDim,nFold);   
    for j=1:SMFAMaxCommonDim
        for i=1:nFold
            
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.SMFA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.SMFA.TestRates{j}.TotalRate;
          
        end

    CrossValid.Total.SMFA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.SMFA{j}.TrainRateVar = var(TrRates(j,:));

    CrossValid.Total.SMFA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.SMFA{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%SDMFA
I = find(strcmp(MethodsOn,'SDMFA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(SDMFAMaxCommonDim,nFold);    
    TsRates = zeros(SDMFAMaxCommonDim,nFold);   
    for j=1:SDMFAMaxCommonDim
        for i=1:nFold
            
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.SDMFA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.SDMFA.TestRates{j}.TotalRate;
          
        end

    CrossValid.Total.SDMFA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.SDMFA{j}.TrainRateVar = var(TrRates(j,:));

    CrossValid.Total.SDMFA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.SDMFA{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%SDA
I = find(strcmp(MethodsOn,'SDA')==1);
if(isempty(I)==0)
    
    TrRates = zeros(SDAMaxCommonDim,nFold);    
    TsRates = zeros(SDAMaxCommonDim,nFold);   
    for j=1:SDAMaxCommonDim
        for i=1:nFold
            
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.SDA.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.SDA.TestRates{j}.TotalRate;
          
        end

    CrossValid.Total.SDA{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.SDA{j}.TrainRateVar = var(TrRates(j,:));

    CrossValid.Total.SDA{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.SDA{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

%SGE
I = find(strcmp(MethodsOn,'SGE')==1);
if(isempty(I)==0)
    
    TrRates = zeros(SGEMaxCommonDim,nFold);    
    TsRates = zeros(SGEMaxCommonDim,nFold);   
    for j=1:SGEMaxCommonDim
        for i=1:nFold
            
            TrRates(j,i) = CrossValid.Folds{i}.DRMethods.SGE.TrainRates{j}.TotalRate;
            TsRates(j,i) = CrossValid.Folds{i}.DRMethods.SGE.TestRates{j}.TotalRate;
          
        end

    CrossValid.Total.SGE{j}.TrainRate = mean(TrRates(j,:));
    CrossValid.Total.SGE{j}.TrainRateVar = var(TrRates(j,:));

    CrossValid.Total.SGE{j}.TestRate = mean(TsRates(j,:));
    CrossValid.Total.SGE{j}.TestRateVar = var(TsRates(j,:));

    end
    
end

fprintf('\n')