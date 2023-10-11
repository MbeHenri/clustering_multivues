function result = MVGL(X, k, beta, lambda, islocal_1, islocal_2)
    addpath('./tools');
    
    % X must be a cell of matrix view data where the samples are in column
    X_train = X;
    
    % define value by default
    if nargin < 5
        islocal_1 = 1;
        islocal_2 = 1;
    end;
    if nargin < 6
        islocal_2 = 1;
    end;
    
    if nargin < 3
        beta = 1;
    end;
    
    if nargin < 4
        lambda = 1;
    end;
    
    % number of views
    nv = length(X_train);
    % number of examples
    n = size(X_train{1},2);
    % number of classes
    c = k;
    
    %>>>>>>  Learning for each Single View Graph S(v)
    % initialisation
    S = zeros(n,n,nv);
    Sv = S;
    
    % optimisation
    for v = 1:nv
        %------ update S(v) simultanaly with Q(v) --------
        S(:,:,v) = Updata_Sv(X_train{v},c,k, islocal_1, beta);
        Sv(:,:,v) = S(:,:,v)./nv;
        
    end
    
    %>>>>>>  Learning the global graph A
    % initialisation
    J_old = 1; J_new = 10; EPS = 1e-3;
    iter = 0;
    S0 = sum(Sv,3);
    
    % optimisation
    while abs((J_new - J_old)/J_old) > EPS
        iter = iter + 1;
        
        %------update A simultanaly with P --------
        [A, ~] = Updata_A(S0,c,islocal_2, lambda);
        
        %------ update W --------
        W = Updata_w(A,S);
        for i = 1:n
            for v = 1:nv        
                Sv(:,i,v) = W(v,i).*S(:,i,v);
            end
        end
        S0 = sum(Sv,3);
        clear Ab
        
        J_old = J_new;
        J_new =  sum(sum((A - S0).^2));
        O(iter) = J_new;
    end
    
    %---- compute P ------
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    [P, ~, ev]=eig1(L, c, 0);
    
    
    %---- compute clusters -------
    % Maximum number of iterations for KMeans
    
%{
    MAXiter = 1000; 
    
    % Number of replications for KMeans
    REPlic = 20;
    y = kmeans(P,c,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
      
%}

    %result = {Sv ; A ; P ; y; O};
    result = {Sv ; A ; P ; O};
end