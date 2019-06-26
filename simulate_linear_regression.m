function [X,y,beta] = simulate_linear_regression(nObs,nVars,nActive,cor,signalNoiseRatio)
%--------------------------------------------------------------------------
% simulate_linear_regression.m: 
%--------------------------------------------------------------------------
%
% DESCRIPTION: Simulate data from a linear regression model of the form :
%                   y = X*beta+noise
%
% USAGE:
%    [X,y,beta] = simulate_linear_regression(nObs,nVars,nActive,cor,signalNoiseRatio)
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
% signalNoiseRatio  Signal to noise ratio requested, positive scalar             
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
% DATE: 29 Oct 2015
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
%    24 Nov 2015: Original version of simulate_linear_regression.m written.
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
    X = sqrt(nObs)*bsxfun(@rdivide, X, sqrt(sum(X.^2, 1)));
    % Columns are renormalized to have unit Eucledian norm
%    X = bsxfun(@rdivide, X, sqrt(sum(X.^2, 1)));

    % Generation of the regression vector beta
    beta = zeros(nVars, 1);
    support = randsample(1:nVars, nActive); 

    beta(support) = 2*binornd(1, .5*ones(nActive, 1)) - 1;
    
    % Rescale regression vector to given signal to noise ratio
    % such that \|X*beta\|^2_2/nObs = signalNoiseRatio

    beta = sqrt(signalNoiseRatio*nObs/(norm(X*beta, 2)^2)) * beta; 

    % Generation of the noise vector noise
    % iid entries from a normal distribution with mean 0 and variance 1

    Sigma_noise = 1;
    mu_noise = 0;
    
    noise = mvnrnd(mu_noise, Sigma_noise, nObs);
    
    % Generation of the output vector y
    y = X*beta + noise;
end
%------------------------------------------------------------------
% End function simulate_linear_regression
%------------------------------------------------------------------
