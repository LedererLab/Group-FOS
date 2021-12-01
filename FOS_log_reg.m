function [betaFOS,lambdaFOS,suppFOS] = FOS_log_reg(X,y,Lambda,c,z)
%--------------------------------------------------------------------------
% log-reg-FOS.m: 
%--------------------------------------------------------------------------
%
% DESCRIPTION: Perform Feature Selection with the logistice-regression and 
%              parameter using FOS scheme. The log-reg problem is solved with 
%              spams. 
%   
% (LASSO) :  minimize ...
%
% USAGE:
%    [betaFOS,lambdaFOS,suppFOS] = FOS-log-reg(X,y,Lambda,c,z)
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
% c           Positive scalar
% z           Positive scalar (recommended )
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
% DATE: june 2020
%
% AUTHORS:
%    Algorithm was designed by AUTHORS
%    
%
% REFERENCES:
%    AUTHORS (2016) Efficient Feature Selection
%    	with Large and High-Dimensional Data
%    
%
% 
%
% EXAMPLES:
%
%
%
%
% OLDER UPDATES:   
    if nargin < 3
        error('More input arguments needed.');
    end
    
    if nargin < 4
       % c = 0.75;
    end

    if nargin < 5
       %  a = 1;
    end
    param.loss='logistic';
	param.regul='l1';
%     param.subgrad=true;
    param.max_it=5;
    Lambda = sort(Lambda,'descend');
    M = length(Lambda);
   [nobs,nvars] = size(X);
    objFunc_tol=1e-5;
    temp=zeros(4,1);
    % Initialization
    statsCont = true;
    statsIt = 1;
    Beta = zeros(nvars,M);
    lambdaFOS = Lambda(end);
    
   
    
    iter = zeros(1,M);
    while(statsCont && statsIt<M)
        
        statsIt = statsIt+1;
                
        lambdaCur = Lambda(statsIt);

        
        stopCrit = false;
        
        betaOld = Beta(:,statsIt-1);        

	    stopThresh =z*nobs*(lambdaCur)^2*(c-(1/(z*nobs)))^2; % stopping threshold for the gradian
        param.lambda=lambdaCur;
        
        [betaOld , optim_info]=mexFistaFlat(y, X, betaOld, param);
        Beta(:,statsIt)=betaOld;
        it_num=2;
          while(stopCrit==false)
          gradls=gradcal(X,y,betaOld);
          temp1=sign(betaOld);  temp1(temp1==0)=(-1/lambdaCur)*(gradls(temp1==0)); temp2=sign(betaOld); temp2(temp2==0)=-1;  temp3=sign(betaOld);%set the value of subdifrential in 0 coordinates
        

%          grad3= norm( gradls+(lambdaCur* temp3.'));%assign the  value of grad in the current beta
%          grad2= norm( gradls+(lambdaCur* temp2.'));%assign the  value of grad in the current beta
           grad1= norm( gradls+(lambdaCur* temp1.'));%assign the  value of grad in the current beta
%          if ( grad1 <= stopThresh || grad2 <= stopThresh || grad3 <= stopThresh)
%              flag=1;
%          end
% % %    
	    if ( grad1 <= stopThresh  )% stopping criterion
                Beta(:,statsIt)=betaOld;
                stopCrit = true;
        else
                temp=optim_info;
                [betaOld , optim_info]=mexFistaFlat(y, X, betaOld, param);
                it_num=it_num+2;
        end
        end

        % Statistical test
        
        statsCont = all(max(abs(bsxfun(@minus, Beta(:,statsIt), Beta(:,1:statsIt))),[],1) ./ (Lambda(statsIt)+Lambda(1:statsIt))- 2*c <= 0);
    end
    
    if statsCont == false 
        betaFOS = Beta(:,statsIt-1);
        lambdaFOS = Lambda(statsIt-1);
    else
        betaFOS = betaOld;
    end
	
    % Thresholding    
    suppFOS = find(abs(betaFOS) >= 6*c*lambdaFOS);
    
end
%------------------------------------------------------------------
% End function lassoFOS
%------------------------------------------------------------------
