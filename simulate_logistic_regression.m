function [X,y,beta] = simulate_logistic_regression(nObs,nVars,nActive,cor,signalLevel)
%--------------------------------------------------------------------------
% simulate_logistic_regression.m: 
%--------------------------------------------------------------------------
%
% DESCRIPTION: Simulate data from a logistic regression model of the form :
%                   Pr(y=1|x) = exp(x'*beta)/(1+exp(x'*beta))
% USAGE:
%    [X,y,beta] = simulate_logistic_regression(nObs,nVars,nActive,cor,signalLevel)
%
%
% EXTERNAL FUNCTIONS:
%
% INPUT ARGUMENTS:
% 
% nObs        Number of observations, positive integer
% nVars       Number of variables, positive integer
% nActive     Size of the support of beta, positive integer at most nVars
% cor         Magnitude of mutual correlations, scalar with 0 <= cor < 1
% signalLevel  Desired signal level for coefficients in support, positive scalar             
%
%
% OUTPUT ARGUMENTS:
% X           Input matrix, of dimension nObs x nVars; each row is an
%             observation vector.
% y           Response variable, vector of dimension nObs x 1. 
% beta        Regression vector of dimension nVars x 1.
%
% DETAILS:
%
%
% LICENSE: 
%
% DATE: 2 June 2020
%
% AUTHORS:
%    Algorithm was designed by Nehemy Lim and Johannes Lederer
%    Department of Statistics, University of Washington, USA.
%
% REFERENCES:
%    Lederer et al. (2015) A practical scheme and fast algorithm to tune
%    the Lasso with optimal guarantees
%
%
% SEE ALSO:
%    
%
% EXAMPLES:
%
%
% DEVELOPMENT:
%    2 June 2020: Original version of simulate_logistic_regression.m written.
%
%
% OLDER UPDATES:     

    if nargin < 5
        error('More input arguments needed.');
    end    

    % Generation of the design matrix X
    % iid distributed rows from a normal distribution with equicorrelated design

    Sigma_X = (1-cor)*eye(nVars) + cor*ones(nVars);
    mu_X = zeros(1, nVars);
    
    X = mvnrnd(mu_X, Sigma_X, nObs);

    % Columns are renormalized to have Eucledian norm exactly sqrt(nObs)
%     X = sqrt(nObs)*bsxfun(@rdivide, X, sqrt(sum(X.^2, 1)));
    % Columns are renormalized to have unit Eucledian norm
%     X = bsxfun(@rdivide, X, sqrt(sum(X.^2, 1))); 
%    X=mexNormalize(X);
%     X=X-repmat(mean(X),[size(X,1) 1]); X=mexNormalize(X);
    % Generation of the regression vector beta
    beta = zeros(nVars, 1);
    support = randsample(1:nVars, nActive); 

%     beta(support) = 2*binornd(1, .5*ones(nActive, 1)) - 1;
    
    % Regression coefficients in the support are drawn from
    % a normal distribution with mean signalLevel and variance 1

      beta(support) = (mvnrnd(signalLevel, 1, nActive));
%       beta(support) = 2*binornd(1, .5*ones(nActive, 1)) - 1;
        
    % Get probabilities for each observation
    probs = 1./(1+exp(-X*beta));
    
    % Generation of the output vector y
    y = binornd(1, probs);  
    y(y==0)=-1;
    
end
%------------------------------------------------------------------
% End function simulate_logistic_regression
%------------------------------------------------------------------