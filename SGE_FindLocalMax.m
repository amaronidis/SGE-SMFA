function [LocalMax,Ids] = SGE_FindLocalMax(A)

%FIND LOCAL MAX

%This function locates the local maxima of a row vector 
%
%Input->            A: A row vector the local maxima of which we are looking for
%
%Outputs->          LocalMax: A row vector with the local maxima
%                   Ids:      The corresponding indices

LocalMax = [];
Ids      = [];

s = length(A);

if (s==1)
    
    LocalMax = A(1);
    Ids      = 1;
    
else

    %Boundary checking for local maximum
    if A(1)>A(2)    

        LocalMax = [LocalMax A(1)];

        Ids      = [Ids 1];

    end

    if (s>2)

    for i=2:(s-1)

        if (A(i)>A(i-1) && A(i)>A(i+1))

        %Internal checking for local maxima
        LocalMax = [LocalMax A(i)];

        Ids      = [Ids i];

        end

    end

    end

    %Boundary checking for local maximum
    if A(s)>A(s-1)   

        LocalMax = [LocalMax A(s)];

        Ids      = [Ids s];

    end

end