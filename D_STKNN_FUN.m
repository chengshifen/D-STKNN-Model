%% ST KNN model
% sample_in    Num*(m*n) 
% sample_out   Num*step  
% input_data   N*(m*n)   
function result = D_STKNN_FUN(sample_in,sample_out,input_data,param,corr_vector)
%% 1. determine size of input
[N,L]=size(input_data);
[Num,mf] = size(sample_out); % mf is the maximum forecast step

%% 2. init parameter
m = param(1); n = param(2);  a3 = param(3); k = param(4);

Wt = zeros(m,m); %time weight
Ws = zeros(n,n); %space weight

for i=1:m
    Wt(i,i)=(i/sum(1:m));%<Eq. (15)> in this paper
end
for i=1:n
    Ws(i,i)=(corr_vector(i)/sum(corr_vector));       %<Eq. (16)> in this paper
end

%% 3. forecasting process
F = zeros(N,mf);% init forecasting result with scale 
for i =1:N
    SD = zeros(Num,1); %  init similarity vector
    for j = 1:Num
        V = reshape(input_data(i,:),n,m)'; % current state
        Vp = reshape(sample_in(j,:),n,m)'; % historical state
        SD(j,1) = sqrt(sum(sum((Wt*(V-Vp)*Ws).^2)));%similarity between V and Vpm,  <Eq. (17)> in this paper
    end
    Index = 1:Num;
    Sel = sortrows([SD,Index';],1); %Sort the SD to choose the most similar k samples
    lamda1 = 1/(4*pi*a3^2)*exp(-1/(4*a3^2)*Sel(1:k,1).^2);%<Eq. (19)>  in this paper
    for j = 1:mf
        F(i,:) = sum(lamda1*ones(1,mf).*sample_out(Sel(1:k,2),:))/sum(lamda1);% <Eq. (18)> in this paper
    end
end
result = F;




