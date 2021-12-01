% addpath(genpath('../glmnet'))
%clear all;
%% Simulation of data from logistic regression models
nObs=200; %number of data examples
nVarsVec=200;% number of variables
nVarsVecSize = length(nVarsVec);
nActive=8;  %number of activated variables
method='logFOS'; 
corVec=0.5;

corVecSize = length(corVec);
signalNoiseRatio=5;
signallevel=2;
cStats =6; % c_log in the paper
cComp = 1; % z in the paper


fracMax = 1e-4; % fraction of largest tuning parameter
% Careful ! standardize = false if data already standardized
options_glmnet.standardize = false;
options_glmnet.intercept = false;
% Number of folds for cross-validation
Kfold = 10;
nRun = 10;
% Hamming distance means and standard deviations

hamming_dist_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dis_LI_std=zeros(nVarsVecSize*corVecSize,2);
hamming_dis_lassospams_std=zeros(nVarsVecSize*corVecSize,2);
% Run time means and standard deviations

time_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_LI_std=zeros(nVarsVecSize*corVecSize,2);
time_lassospams_std=zeros(nVarsVecSize*corVecSize,2);

estim_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);
estim_log_lasso_mean_std=zeros(nVarsVecSize*corVecSize,2);
estim_LI_mean_std=zeros(nVarsVecSize*corVecSize,2);

for iVars=1:nVarsVecSize
    for iCor=1:corVecSize       
        
        
        hamming_dist_FOS_ista_tmp = zeros(1, nRun);
        hamming_dist_LI_tmp= zeros(1, nRun);
        hamming_dist_lassospams_tmp= zeros(1, nRun);


        
        time_FOS_ista_tmp = zeros(1, nRun);
        time_LI_tmp=zeros(1,nRun);
        time_lassospams_tmp=zeros(1,nRun);
        
        estim_Log_FOS = zeros(1, nRun);
        estim_LI = zeros(1, nRun);
        estim_Log_lasso = zeros(1, nRun);
        for iRun=1:nRun
            rng((iRun+3));           
            [X_unstd, y_unstd, betaTrue_unstd] = simulate_logistic_regression(nObs,nVarsVec(iVars),nActive,corVec(iCor),signallevel);
            suppTrue = (betaTrue_unstd ~= 0);
            X=X_unstd;
            y=y_unstd;
            lambda_max = 10*(log(nVarsVec)/nObs);
             %lambda_max=0.001;
             lambdaVec_FOS = sort(linspace(fracMax*lambda_max, lambda_max, 500), 'descend');           
             tic
            [betaFOS,lambdaFOS,suppFOSind] = FOS_log_reg(X,y,lambdaVec_FOS,cStats,cComp);
	        time_FOS_ista_tmp(iRun) = toc;
            suppFOS = zeros(nVarsVec(iVars), 1);
	        suppFOS(suppFOSind) = 1;
            hamming_dist_FOS_ista_tmp(iRun) = sum(suppFOS ~= suppTrue);
            estim_Log_FOS(iRun)=norm(betaFOS-betaTrue_unstd,Inf);
           

            
%%%%%%%%%%%%loglassoSpams
          varError = 1;
          tic;
%          
%           [beta_CV] = cv_loglasso_fista_v2(X,y,lambdaVec_FOS,Kfold);
           beta_CV=zeros(nVarsVec(iVars),1);
          time_lassospams_tmp(iRun) = toc;
          supplassospams = (beta_CV~=0);
          hamming_dist_lassospams_tmp(iRun) = sum(supplassospams ~= suppTrue);  
           estim_Log_lasso(iRun)=norm(beta_CV-betaTrue_unstd,Inf);
%%%%%%%%%%%%%%%%%%% LI2019
%             tic
%             [betaLI,lambdaLI,suppLIindLI] = Li_log_reg(X,y,lambdaVec_FOS,cStats,cComp);
% 	        time_LI_tmp(iRun) = toc;
%             suppLI = zeros(nVarsVec(iVars), 1);
% 	        suppLI(suppLIindLI) = 1;
%             hamming_dist_LI_tmp(iRun) = sum(suppLI ~= suppTrue);
%             estim_LI(iRun)=norm(betaLI-betaTrue_unstd,Inf);
             end
       
        % Hamming distance means
        
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_FOS_ista_tmp);
%          hamming_dis_LI_std(corVecSize*(iVars-1)+iCor, 1) = mean( hamming_dist_LI_tmp);
         hamming_dis_lassospams_std(corVecSize*(iVars-1)+iCor, 1) = mean( hamming_dist_lassospams_tmp);

        
        % Hamming distance standard deviations
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_FOS_ista_tmp);
%          hamming_dis_LI_std(corVecSize*(iVars-1)+iCor, 2) = std( hamming_dist_LI_tmp);
         hamming_dis_lassospams_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_lassospams_tmp);

        % Mean run times
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_FOS_ista_tmp);
%          time_LI_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_LI_tmp);
         time_lassospams_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_lassospams_tmp);

        % Standard deviation run times
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_FOS_ista_tmp); 
%          time_LI_std(corVecSize*(iVars-1)+iCor, 2) = std(time_LI_tmp);
         time_lassospams_std(corVecSize*(iVars-1)+iCor, 2) = std(time_lassospams_tmp);
         
         estim_FOS_ista_mean_std (corVecSize*(iVars-1)+iCor, 1) = mean(estim_Log_FOS);
         estim_log_lasso_mean_std (corVecSize*(iVars-1)+iCor, 1) = mean(estim_Log_lasso);
         estim_FOS_ista_mean_std (corVecSize*(iVars-1)+iCor, 2) = std(estim_Log_FOS);
         estim_log_lasso_mean_std (corVecSize*(iVars-1)+iCor, 2) = std(estim_Log_lasso);
%          estim_LI_mean_std (corVecSize*(iVars-1)+iCor, 2) = std(estim_LI);
%          estim_LI_mean_std (corVecSize*(iVars-1)+iCor, 1) = mean(estim_LI);



    end
end
sdate_str = date;

fprintf('\n log-Fos_HamDis=%4.2f, Var_HamDis=%4.2f', hamming_dist_FOS_ista_mean_std(1,1), hamming_dist_FOS_ista_mean_std(1,2))
fprintf('\n log-Fos_EstErr=%4.2f, Var_EstErr=%4.2f',  estim_FOS_ista_mean_std (1,1), estim_FOS_ista_mean_std (1,2))
fprintf('\n log-Fos_RunTime=%4.2f, Var_RunTime=%4.2f ',  time_FOS_ista_mean_std(1,1),time_FOS_ista_mean_std(1,2))


