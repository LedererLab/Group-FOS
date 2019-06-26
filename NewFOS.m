function [betaFOS,lambdaFOS,suppFOS,betaFOS_sparse] = NewFOS(X,y,Lambda,cStats,cComp,gamma,groups)
%--------------------------------------------------------------------------
% lassoFOS.m: 
%--------------------------------------------------------------------------
%
% DESCRIPTION: Perform Feature Selection with the Lasso and 
%              parameter using FOS scheme. The LASSO problem is solved with 
%              a proximal gradient type of approach (FISTA).
%   
% (LASSO) :  minimize f(beta)=||y-X*beta||_2^2 + lambda*||beta||_1
%
% USAGE:
%    [betaFOS,lambdaFOS,suppFOS] = NewFOS(X,y,Lambda,cStats,cComp,gamma)
%
%
% EXTERNAL FUNCTIONS:
%
% INPUT ARGUMENTS:
% X           Input matrix, of dimension nobs x nvars; each row is an
%             observation vector. Can be in sparse matrix format.
% 			  All the columns of X should have mean 0 and l2-norm sqrt(nobs)!
% y           Response variable, vector of dimension nobs x 1. 
%  			  y should have mean 0 and unit l2-norm !
% Lambda      Vector of positive regularization parameters.
% cStats      Positive scalar
% cComp       Positive scalar (recommended between 0 and 1)
%             
% OUTPUT ARGUMENTS:
% betaFOS       Regression vector (a vector of length nvars x 1)
% lambdaFOS     Selected regularization parameter
% suppFOS 	Vector of indices of variables in the estimated support
%
% DETAILS:
%
%
% LICENSE: 
%
% DATE: June 2019
%
% AUTHORS:
%    Algorithm was designed by AUTHORS
%    
%
%    
%
% SEE ALSO:
%    Require the SPAMS toolbox : function mexFistaFlat.
%
% EXAMPLES:
%
%
% DEVELOPMENT:
%     2019
%
% OLDER UPDATES:   
    if nargin < 3
        error('More input arguments needed.');
    end
    
    if nargin < 4
        cStats = 2;
    end

    if nargin < 5
        cComp = 1;
    end
     
    if nargin < 6
        gamma = 1;
    end

    Lambda = sort(Lambda,'descend');
    M = length(Lambda);
    [nobs,nvars] = size(X);
    normYsq = norm(y, 2)^2;

    % Initialization
    statsCont = true;
    statsIt = 1;
    Beta = zeros(nvars,M);
    lambdaFOS = Lambda(end);
    
    %Setting FISTA parameters
	%param.ista = true; % use ista instead of fista, false by default
	param.loss='square';
	param.regul='l1';
    param. it0=1;
    ngroups=max(groups);
    iter = zeros(1,M);
    while(statsCont && statsIt<M)
        
        statsIt = statsIt+1;
                
        lambdaCur = Lambda(statsIt);
% 		mexFistaFlat solves : minimize 0.5*||y-X*beta||_2^2 + lambda*||beta||_1
        param.lambda = 0.5*lambdaCur;
        
        stopCrit = false;
        
        betaOld = Beta(:,statsIt-1);        
     
 	   stopThresh = 0.5*gamma * cComp^2*(lambdaCur)^2/(nobs*(cStats^2));  % stopping threshold for the duality gap
	     
       param.tol=stopThresh;
       betaOld = mexFistaFlat(y, X, betaOld, param);
       Beta(:,statsIt)=betaOld;

        % Statistical test
        statsCont = all(max(abs(bsxfun(@minus, Beta(:,statsIt), Beta(:,1:statsIt))),[],1) ./ (Lambda(statsIt)+Lambda(1:statsIt)) - (3/(2*nobs*cStats)) <= 0);
       
    end
    
    if statsCont == false 
        betaFOS = Beta(:,statsIt-1);
        lambdaFOS = Lambda(statsIt-1);
    else
        betaFOS = betaOld;
    end
	
    % Thresholding    
    suppFOS = find(abs(betaFOS) >= ((9*lambdaFOS) / (2*nobs*cStats)));
    betaFOS_sparse=zeros(nvars,1);
    betaFOS_sparse( suppFOS,1)=betaFOS(suppFOS,1);
end

