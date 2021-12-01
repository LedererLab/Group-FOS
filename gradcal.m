function [grad] = gradcal(X,Y,beta)
[n,p]=size(X);
temp=zeros(1,p); grad=0;
for i=1:n
    temp=temp+Y(i,1)*X(i,:)*sigmoid_func(X(i,:),Y(i,1),beta);
end
  grad=temp/(-1*n);
end

function [sig_value] = sigmoid_func(x,y,beta)
sig_value=exp(-y*(x*beta))/(1+exp(-y*(x*beta)));
end

