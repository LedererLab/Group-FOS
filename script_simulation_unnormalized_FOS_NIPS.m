addpath(genpath('../glmnet'))
clear all;
%% Simulation of data from linear regression models
nObs=500;%500;%
nVarsVec=1000;%1000;%
nVarsVecSize = length(nVarsVec);
nActive=10;%10;% number of activated variables
method='fos'; % 2 values 1-'groupfos' or  2-'fos': in case of groupfos you should set the value of groups_l
groups_l=10; %lenght of groups 
groups_n=round(nVarsVec/groups_l);  %number of groups
groups=zeros(1,nVarsVec);
corVec=0.3;%[0 0.5]
corVecSize = length(corVec);
signalNoiseRatio=5;%0;%
%make the index of groups
for k=1:groups_n
   groups(1,1+(k-1)*groups_l:(1+(k-1)*groups_l)+groups_l)=k;
end
groups=int32(groups(1,1:nVarsVec));
cStats =2; % Careful ! May need to be changed depending on normalization and lasso formulation
cComp = 1;%0.1;% always scale free, not dependent on normalization or lasso formulation
% cStatsAVinf = .75;
fracMax = 1e-3; % fraction of largest tuning parameter
fprintf('cStats=%0.2e cComp=%0.2e\n', cStats, cComp)
% Careful ! standardize = false if data already standardized
options_glmnet.standardize = false;
options_glmnet.intercept = false;
% Number of folds for cross-validation
Kfold = 10;
nRun = 10;%100;% % number of runs for each set of parameters
% Hamming distance means and standard deviations
hamming_dist_CVglmnet_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dist_CV_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dist_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dis_grplasso_std=zeros(nVarsVecSize*corVecSize,2);
hamming_dis_grpFOS_std=zeros(nVarsVecSize*corVecSize,2);
% Run time means and standard deviations
time_CVglmnet_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_CV_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_grplasso_std=zeros(nVarsVecSize*corVecSize,2);
time_grpFOS_std=zeros(nVarsVecSize*corVecSize,2);
for iVars=1:nVarsVecSize
    for iCor=1:corVecSize       
        
        hamming_dist_CVglmnet_tmp = zeros(1, nRun);
        hamming_dist_CV_tmp = zeros(1, nRun);
        hamming_dist_FOS_ista_tmp = zeros(1, nRun);
        hamming_dist_NewFOS_ista_tmp = zeros(1, nRun);
        hamming_dist_grplasso= zeros(1, nRun);
        hamming_dist_grpFOS_tmp= zeros(1, nRun);

        time_CVglmnet_tmp = zeros(1, nRun);
        time_CV_tmp = zeros(1, nRun);
        time_FOS_ista_tmp = zeros(1, nRun);
        timegroup_FOS=zeros(1,nRun);
        timegroup_lasso=zeros(1,nRun);

        for iRun=1:nRun
             rng((iRun+3));
              switch (method)
                case 'groupfos'
            [X_unstd, y_unstd, betaTrue_unstd] = simulate_linear_regression_group(nObs,nVarsVec(iVars),nActive,corVec(iCor),signalNoiseRatio);
            suppTrue = (betaTrue_unstd ~= 0);
            X=X_unstd;
            y=y_unstd;
            lambda_max = 2*norm(X'*y, Inf);
            % formulation of lasso in FOS : \|y-X*beta\|_2^2 + lambda*\|beta\|_1
%            lambdaVec_FOS = sort(lambda_max./1.3.^(0:99), 'descend');
            lambdaVec_FOS = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 100), 'descend');
            tic
            [betagroup_FOS,lambdagr_FOS,suppFOS_group] = GroupFOS(X,y,lambdaVec_FOS,cStats,cComp,1,groups);
            timegroup_FOS(iRun)=toc;
            suppgroup_FOS=zeros(nVarsVec,1);
            suppgroup_FOS(ismember(groups,suppFOS_group))=1;
            hamming_dist_grpFOS_tmp(iRun) = sum(suppgroup_FOS ~= suppTrue);

             case 'fos'
            [X_unstd, y_unstd, betaTrue_unstd] = simulate_linear_regression(nObs,nVarsVec(iVars),nActive,corVec(iCor),signalNoiseRatio);
            suppTrue = (betaTrue_unstd ~= 0);
            X=X_unstd;
            y=y_unstd;
            lambda_max = 2*norm(X'*y, Inf);
            % formulation of lasso in FOS : \|y-X*beta\|_2^2 + lambda*\|beta\|_1
%            lambdaVec_FOS = sort(lambda_max./1.3.^(0:99), 'descend');
            lambdaVec_FOS = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 100), 'descend');
            tic
            [betaFOS,lambdaFOS,suppFOSind] = NewFOS(X,y,lambdaVec_FOS,cStats,cComp,1,groups);
	        time_FOS_ista_tmp(iRun) = toc;
            suppFOS = zeros(nVarsVec(iVars), 1);
	        suppFOS(suppFOSind) = 1;
            hamming_dist_FOS_ista_tmp(iRun) = sum(suppFOS ~= suppTrue);
            end %switch
            % formulation of lasso in glmnet : (1/(2*nObs))*\|y-X*beta\|_2^2 + lambda*\|beta\|_1            
            lambda_max = norm(X'*y, Inf)/nObs;
%	        options_glmnet.lambda = sort(lambda_max./1.3.^(0:99), 'descend');
            options_glmnet.lambda = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 100), 'descend');
            tic;
%            CVerr = cvglmnet(X,y,[],options_glmnet);
%            beta_CVglmnet = CVerr.glmnet_fit.beta(:, CVerr.lambda == CVerr.lambda_min);
%            idxLambdaGlmnet = find(CVerr.lambda == CVerr.lambda_min);
%            nnzGlmnet = nnz(beta_CVglmnet);
 	         beta_CVglmnet = zeros(nVarsVec(iVars),1);
%  	        time_CVglmnet_tmp(iRun) = toc;
            suppCVglmnet = (beta_CVglmnet~=0);
            hamming_dist_CVglmnet_tmp(iRun) = sum(suppCVglmnet ~= suppTrue);                          
            tic
%             result_grplasso=grplasso(X,y,lambdaVec_FOS,Kfold,groups);
%             timegroup_lasso(iRun)=toc;
%             hamming_dist_grplasso(iRun)=sum(result_grplasso.suppbeta~=suppTrue);
            tic
                 
             end
       
        % Hamming distance means
         hamming_dist_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_CVglmnet_tmp);
         hamming_dist_CV_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_CV_tmp);
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_FOS_ista_tmp);
%          hamming_dis_grplasso_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_grplasso);
         hamming_dis_grpFOS_std(corVecSize*(iVars-1)+iCor, 1) = mean( hamming_dist_grpFOS_tmp);
        
        % Hamming distance standard deviations
         hamming_dist_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_CVglmnet_tmp);
         hamming_dist_CV_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_CV_tmp);
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_FOS_ista_tmp);
%          hamming_dis_grplasso_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_grplasso);
         hamming_dis_grpFOS_std(corVecSize*(iVars-1)+iCor, 2) = std( hamming_dist_grpFOS_tmp);
        % Mean run times
         time_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_CVglmnet_tmp);
         time_CV_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_CV_tmp);
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_FOS_ista_tmp);
%          time_grplasso_std(corVecSize*(iVars-1)+iCor, 1) = mean(timegroup_lasso);
         time_grpFOS_std(corVecSize*(iVars-1)+iCor, 1) = mean(timegroup_FOS);
        % Standard deviation run times
         time_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_CVglmnet_tmp);
         time_CV_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_CV_tmp);
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_FOS_ista_tmp); 
%          time_grplasso_std(corVecSize*(iVars-1)+iCor, 2) = std(timegroup_lasso);
         time_grpFOS_std(corVecSize*(iVars-1)+iCor, 2) = std(timegroup_FOS);
    end
end
sdate_str = date;

% save(sprintf('simulated_lin_reg_unnormalized_NIPS_%s_nVars%d_%d_cor%0.2f_%0.2f_SNR_%d.mat', ... 
%    date_str, nVarsVec(1), nVarsVec(end), corVec(1), corVec(end), signalNoiseRatio), ...
%    'corVec', 'nVarsVec', ...
%    'hamming_dist_CVglmnet_mean_std', ...
%    'hamming_dist_CV_mean_std', ...
%    'hamming_dist_FOS_ista_mean_std', ...
%    'time_CVglmnet_mean_std', ...
%    'time_CV_mean_std', ...
%    'time_FOS_ista_mean_std') 
