function MappedX = SGE_Projection(X,dim,TransMatrix)

%PROJECTION

%This function projects a dataset into a new space by using the Transform Matrix

%Inputs->       X:           Data Matrix
%               dim:         Desired column-ids of TransMatrix to keep->
%                                                   (dim=0):            Keeps all columns
%                                                   (dim=[a b ... z]):  ids of columns to keep
%               TransMatrix: Transformation Matrix
%
%Outputs->      MappedX:     Mapped Data Matrix

if(dim==0)

MappedX = TransMatrix' * X;

else
    
TM = TransMatrix(:,dim);

MappedX = TM' * X;

end