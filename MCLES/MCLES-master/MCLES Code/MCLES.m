function result = MCLES(X, k, alpha, beta, d, gamma, maxIters)
    % number of views
    V = size(X,2);
    % number of examples
    N = size(X{1},2);
    % number of classes
    C = k;
    
    % normalisation by column of matrix view data
    for i=1:V
        X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);  %normalized
    end
    
    % dimension of each view
    for i=1:V
        D{i} = size(X{i},1); 
    end
    SD = 0;
    
    % contatenation of matrix view data
    M = [];
    for i=1:V
        SD = SD + D{i};
        M = [M;X{i}]; %X
    end
    
    W = zeros(SD,d);
    S = zeros(N);
    
    % initialisation
    H = rand(d,N);
    F = rand(N,C);
    
    % optimisation
    for it=1:maxIters
        
        %>>>> learning of latent embedding space
        %------update W--------
        W = UpdateW(H,M,W);
        %------update H--------
        H = SMR_mtv(M,W,S,alpha);
        
        %>>>> learning of global similarity matrix
        %------update S--------
        S = UpdateS(H'*H,F,beta/alpha,gamma/alpha);
        
        %>>>> learning of cluster indicator matrix
        %------update F--------
        Z = S;
        Z= (Z+Z')/2;
        D = diag(sum(Z));
        L = D-Z;
        [F, ~, ~]=eig1(L, C, 0);
        
        %------analyse convergence-------
        Obj(it) = norm((M-W*H),'fro')^2+alpha*norm((H-H*S),'fro')^2+beta*norm(S,'fro')^2+gamma*trace(F'*L*F);
        if (it>1 && (abs(Obj(it)-Obj(it-1))/Obj(it-1)) < 10^-2)
            break;
        end
    end
    
    
    %---- compute clusters -------

%{
     % Maximum number of iterations for KMeans
    MAXiter = 1000; 
    % Number of replications for KMeans
    REPlic = 20;
    y = kmeans(F,C,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    
    result = {W; H; S; F; y; Obj}; 
%}

    
    result = {W; H; S; F; Obj};
    
end