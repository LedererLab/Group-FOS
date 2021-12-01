% addpath(genpath('glmnet_matlab-master'))
clear all;
%% Simulation of data from linear regression models
nObs=500; %number of data examples
nVarsVec=1000;% number of varibles
nActive=10; % number of activated variables
method='fos'; % 2 values 1:'fos' ;  2-'groupfos': in case of groupfos we should also set the value of groups_l
groups_l=5; %lenght of groups is used once we work we groupfos

nVarsVecSize = length(nVarsVec);
groups_n=round(nVarsVec/groups_l);  %number of groups
groups=zeros(1,nVarsVec);
corVec=0.3;
corVecSize = length(corVec);
signalNoiseRatio=5;
%make the index of groups
for k=1:groups_n
   groups(1,1+(k-1)*groups_l:(1+(k-1)*groups_l)+groups_l)=k;
end
groups=int32(groups(1,1:nVarsVec));
cStats =2; %parameter "c" in the paper
cComp = 1; % parameter "z" in the paper

fracMax = 1e-3; % fraction of largest tuning parameter
% Careful ! standardize = false if data already standardized
options_glmnet.standardize = false;
options_glmnet.intercept = false;
% Number of folds for cross-validation
Kfold = 10;
nRun = 10;  % number of runs for each set of parameters
% measure of accuracies
hamming_dist_CVglmnet_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dist_CV_mean_std = zeros(nVarsVecSize*corVecSize,2);
hamming_dist_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2); %keeps the hamming distance of FOS
hamming_dis_grplasso_std=zeros(nVarsVecSize*corVecSize,2);
hamming_dis_grpFOS_std=zeros(nVarsVecSize*corVecSize,2);            %keeps the hamming distance of groupFOS
hamming_dis_lassospams_std=zeros(nVarsVecSize*corVecSize,2);
hamming_dist_CHi_std = zeros(nVarsVecSize*corVecSize,2);
esstim_error_CVglmnet_mean_std = zeros(nVarsVecSize*corVecSize,2);
esstim_error_lassospams_mean_std = zeros(nVarsVecSize*corVecSize,2);
esstim_error_FOS_mean_std = zeros(nVarsVecSize*corVecSize,2);       %keeps the estimation error of FOS
esstim_error_grLasso_mean_std = zeros(nVarsVecSize*corVecSize,2);
esstim_error_grpFOS_mean_std = zeros(nVarsVecSize*corVecSize,2);    %keeps the estimation error of groupFOS
esstim_error_Chi_std = zeros(nVarsVecSize*corVecSize,2);

% Run time means and standard deviations
time_CVglmnet_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_CV_mean_std = zeros(nVarsVecSize*corVecSize,2);
time_FOS_ista_mean_std = zeros(nVarsVecSize*corVecSize,2);    %keeps the running-time of FOS
time_grplasso_std=zeros(nVarsVecSize*corVecSize,2);
time_grpFOS_std=zeros(nVarsVecSize*corVecSize,2);              %keeps the runnig-time of group-FOS
time_lassospams_std=zeros(nVarsVecSize*corVecSize,2);
time_Chi_std = zeros(nVarsVecSize*corVecSize,2);
for iVars=1:nVarsVecSize
    for iCor=1:corVecSize       
        
        hamming_dist_CVglmnet_tmp = zeros(1, nRun);
        hamming_dist_CV_tmp = zeros(1, nRun);
        hamming_dist_FOS_ista_tmp = zeros(1, nRun);
        hamming_dist_NewFOS_ista_tmp = zeros(1, nRun);
        hamming_dist_grplasso= zeros(1, nRun);
        hamming_dist_grpFOS_tmp= zeros(1, nRun);
        hamming_dist_lassospams_tmp= zeros(1, nRun);
        hamming_dist_CHi_tmp = zeros(1, nRun);
        esstim_error_CVglmnet=zeros(1, nRun);
        esstim_error_FOS=zeros(1, nRun);
        esstim_error_grpFOS=zeros(1, nRun);
        esstim_error_grplassos=zeros(1, nRun);
        esstim_error_lassospams=zeros(1, nRun);
        esstim_error_CHi = zeros(1, nRun);

        time_CVglmnet_tmp = zeros(1, nRun);
        time_CV_tmp = zeros(1, nRun);
        time_FOS_ista_tmp = zeros(1, nRun);
        timegroup_FOS=zeros(1,nRun);
        timegroup_lasso=zeros(1,nRun);
        time_lassospams_tmp=zeros(1,nRun);
        time_CHi_tmp = zeros(1, nRun);
        for iRun=1:nRun
          rng((iRun));    
          switch (method)
           case 'groupfos'
            [X_unstd, y_unstd, betaTrue_unstd] = simulate_linear_regression_group(nObs,nVarsVec(iVars),nActive,corVec(iCor),signalNoiseRatio);
            suppTrue = (betaTrue_unstd ~= 0);
            X=X_unstd;
            y=y_unstd;
            lambda_max = 2*norm(X'*y, Inf);
            lambdaVec_FOS = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 150), 'descend');
            tic
            [betagroup_FOS,lambdagr_FOS,suppFOS_group] = GroupFOS(X,y,lambdaVec_FOS,cStats,cComp,1,groups,groups_l);
            timegroup_FOS(iRun)=toc;
            suppgroup_FOS=zeros(nVarsVec,1);
            suppgroup_FOS(ismember(groups,suppFOS_group))=1;
            hamming_dist_grpFOS_tmp(iRun) = sum(suppgroup_FOS ~= suppTrue);
            esstim_error_grpFOS(iRun)=norm(betagroup_FOS-betaTrue_unstd,Inf);

           case 'fos'
            [X_unstd, y_unstd, betaTrue_unstd] = simulate_linear_regression(nObs,nVarsVec(iVars),nActive,corVec(iCor),signalNoiseRatio);
             suppTrue = (betaTrue_unstd ~= 0);
             X=X_unstd;
             y=y_unstd;
%             X = centerNormalize(X_unstd);
%             y = centerNormalize(y_unstd);
            lambda_max = 2*norm(X'*y, Inf);
            lambdaVec_FOS = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 100), 'descend');
            tic
            [betaFOS,lambdaFOS,suppFOSind] = NewFOS(X,y,lambdaVec_FOS,cStats,cComp,1,groups);
	        time_FOS_ista_tmp(iRun) = toc;
            suppFOS = zeros(nVarsVec(iVars), 1);
	        suppFOS(suppFOSind) = 1;
            hamming_dist_FOS_ista_tmp(iRun) = sum(suppFOS ~= suppTrue);
            esstim_error_FOS(iRun)=norm(betaFOS-betaTrue_unstd,Inf);
            end %switch
            
            %%%%%%%%%%glmnet code
            lambda_max = norm(X'*y, Inf)/nObs;
%	        options_glmnet.lambda = sort(lambda_max./1.3.^(0:99), 'descend');
            options_glmnet.lambda = sort(logspace(log10(fracMax*lambda_max), log10(lambda_max), 100), 'descend');
            tic;
%               CVerr = cvglmnet(X,y,[],options_glmnet);
%              beta_CVglmnet = CVerr.glmnet_fit.beta(:, CVerr.lambda == CVerr.lambda_min);
%            idxLambdaGlmnet = find(CVerr.lambda == CVerr.lambda_min);
%            nnzGlmnet = nnz(beta_CVglmnet);
    	    beta_CVglmnet = zeros(nVarsVec(iVars),1);
  	        time_CVglmnet_tmp(iRun) = toc;
            suppCVglmnet = (beta_CVglmnet~=0);
            hamming_dist_CVglmnet_tmp(iRun) = sum(suppCVglmnet ~= suppTrue);  
            esstim_error_CVglmnet(iRun)=norm(beta_CVglmnet-betaTrue_unstd,Inf);
            
            %%%%%%%%%%%%%%grouplasso
%            tic
%             result_grplasso=grplasso(X,y,lambdaVec_FOS,Kfold,groups);
%             timegroup_lasso(iRun)=toc;
%             hamming_dist_grplasso(iRun)=sum(result_grplasso.suppbeta~=suppTrue);
%             esstim_error_grplassos(iRun)=norm(result_grplasso.beta-betaTrue_unstd,Inf);

%            tic
%%%%%%%%%%%%lassoSpams
          varError = 1;
          tic;          
%          [beta_CV] = cv_lasso_fista_v2(X,y,lambdaVec_FOS,Kfold);
          beta_CV=zeros(nVarsVec(iVars),1);
          time_lassospams_tmp(iRun) = toc;
          supplassospams = (beta_CV~=0);
          hamming_dist_lassospams_tmp(iRun) = sum(supplassospams ~= suppTrue);  
          esstim_error_lassospams(iRun)=norm(beta_CV-betaTrue_unstd,Inf);
%%%%%%%%%%%%%%%%%CHICHI
%           tic
%             [betaCHi,lambdaCHi,suppCHiind] = CHiCHi(X,y,lambdaVec_FOS,cStats,cComp,1,groups);
% 	        time_CHi_tmp(iRun) = toc;
%             suppCHi = zeros(nVarsVec(iVars), 1);
% 	        suppCHi(suppCHiind) = 1;
%             hamming_dist_CHi_tmp(iRun) = sum(suppCHi ~= suppTrue);
%             esstim_error_CHi(iRun)=norm(betaCHi-betaTrue_unstd,Inf);
%%%%%%%%%%%%%%%%%%%
             end
       
        % Hamming distance means
         hamming_dist_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_CVglmnet_tmp);
         hamming_dist_CV_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_CV_tmp);
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_FOS_ista_tmp);
         hamming_dis_grplasso_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_grplasso);
         hamming_dis_grpFOS_std(corVecSize*(iVars-1)+iCor, 1) = mean( hamming_dist_grpFOS_tmp);
         hamming_dis_lassospams_std(corVecSize*(iVars-1)+iCor, 1) = mean( hamming_dist_lassospams_tmp);
%          hamming_dist_CHi_std(corVecSize*(iVars-1)+iCor, 1) = mean(hamming_dist_CHi_tmp);


        
        % Hamming distance standard deviations
         hamming_dist_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_CVglmnet_tmp);
         hamming_dist_CV_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_CV_tmp);
         hamming_dist_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_FOS_ista_tmp);
         hamming_dis_grplasso_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_grplasso);
         hamming_dis_grpFOS_std(corVecSize*(iVars-1)+iCor, 2) = std( hamming_dist_grpFOS_tmp);
         hamming_dis_lassospams_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_lassospams_tmp);
%          hamming_dist_CHi_std(corVecSize*(iVars-1)+iCor, 2) = std(hamming_dist_CHi_tmp);


        % Mean run times
         time_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_CVglmnet_tmp);
         time_CV_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_CV_tmp);
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_FOS_ista_tmp);
         time_grplasso_std(corVecSize*(iVars-1)+iCor, 1) = mean(timegroup_lasso);
         time_grpFOS_std(corVecSize*(iVars-1)+iCor, 1) = mean(timegroup_FOS);
         time_lassospams_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_lassospams_tmp);
%          time_Chi_std(corVecSize*(iVars-1)+iCor, 1) = mean(time_CHi_tmp);


        % Standard deviation run times
         time_CVglmnet_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_CVglmnet_tmp);
         time_CV_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_CV_tmp);
         time_FOS_ista_mean_std(corVecSize*(iVars-1)+iCor, 2) = std(time_FOS_ista_tmp);           time_grplasso_std(corVecSize*(iVars-1)+iCor, 2) = std(timegroup_lasso);
         time_grpFOS_std(corVecSize*(iVars-1)+iCor, 2) = std(timegroup_FOS);
         time_lassospams_std(corVecSize*(iVars-1)+iCor, 2) = std(time_lassospams_tmp); 
%          time_Chi_std(corVecSize*(iVars-1)+iCor, 2) = std(time_CHi_tmp); 

         
          esstim_error_CVglmnet_mean_std (corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_CVglmnet);
          esstim_error_lassospams_mean_std  (corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_lassospams);
          esstim_error_FOS_mean_std  (corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_FOS);
          esstim_error_grLasso_mean_std  (corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_grplassos);
          esstim_error_grpFOS_mean_std  (corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_grpFOS);
%           esstim_error_Chi_std(corVecSize*(iVars-1)+iCor, 1) = mean(esstim_error_CHi); 

          
          esstim_error_CVglmnet_mean_std (corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_CVglmnet);
          esstim_error_lassospams_mean_std  (corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_lassospams);
          esstim_error_FOS_mean_std  (corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_FOS);
          esstim_error_grLasso_mean_std  (corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_grplassos);
          esstim_error_grpFOS_mean_std  (corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_grpFOS);
%           esstim_error_Chi_std(corVecSize*(iVars-1)+iCor, 2) = std(esstim_error_CHi); 


    end
end
sdate_str = date;
fprintf('\n Fos_HamDis=%4.2f, Var_HamDis=%4.2f', hamming_dist_FOS_ista_mean_std(1,1), hamming_dist_FOS_ista_mean_std(1,2))
fprintf('\n Fos_EstErr=%4.2f, Var_EstErr=%4.2f',  esstim_error_FOS_mean_std(1,1), esstim_error_FOS_mean_std(1,2))
fprintf('\n Fos_RunTime=%4.2f, Var_RunTime=%4.2f ',  time_FOS_ista_mean_std(1,1),time_FOS_ista_mean_std(1,2))



