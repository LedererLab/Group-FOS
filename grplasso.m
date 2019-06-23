function [result] = grplasso(X,Y,lambda,kfold,groups)
nobs=size(X,1);
p=size(X,2);
c = cvpartition(nobs,'KFold',kfold) ;
nlambda=size(lambda,2);
error_vec=zeros(1,nlambda);
param.ista = true; % use ista instead of fista, false by default
	param.loss='square';
	param.regul='sparse-group-lasso-l2';
    param.groups=groups;
    param.tol=1e-3;
    param.lambda2 =0;
    beta_ar=zeros(p,nlambda);
for i=1:nlambda
    for j=1:kfold
       trIdx = c.training(j);
       teIdx = c.test(j);
       param.lambda = 0.5*lambda(1,i);
       beta = mexFistaFlat(Y(trIdx,:), X(trIdx,:), zeros(p,1), param);
       error_vec(1,i)=sum(norm((Y(teIdx,:)-(X(teIdx,:)*beta)),2)^2)+error_vec(1,i);
    end
end
[M,I] = min(error_vec);
param.lambda = 0.5*lambda(1,I);
beta = mexFistaFlat(Y, X, zeros(p,1), param);
result.suppbeta=(beta~= 0);
result.lambda=lambda(1,I);
result.beta=beta;
end

