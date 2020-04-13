%Pattern 2 period 3
%Link [11;25;27;29;30]
%Time period [1,26;27,70;71,116;117,197;198,222;223,288]

clc
clear

%% read data
m = 288;
n = 43;
dataMatrix = zeros(30,288*43); 
numTest = 288*10;  

link = [11;25;27;29;30];
dataMatrix = dataMatrix(link,:);

sample_num = size(dataMatrix,2);    
station_num = size(dataMatrix,1);   
day = sample_num/288;
step = 1;  %%One-step

bucket4 = zeros(46*day,station_num);

for t = 1:station_num
    data_index = dataMatrix(t,:);
    data_index = data_index';
    norm_dataMatrix(:,t) = data_index;  
end

%% Construct sample
for d = 1:day
    temp = norm_dataMatrix((d-1)*288+1:d*288,:);
    bucket4((d-1)*46+1:d*46,:) = temp(71:116,:); 
end

%% Compute cross-correlation
stations_opt = zeros(station_num,station_num);
for t = 1:station_num
    corr_station_num1 = 0;
    for k = 1:station_num
        [XCF1,lags1,bounds1]= crosscorr(bucket4(:,t),bucket4(:,k));           
        [V1,I1] = max(abs(XCF1));
        lagDiff1 = lags1(I1); 
        if(lagDiff1<=1)  
            corr_station_num1 = corr_station_num1 + 1;
            station_bucket1(t,corr_station_num1) = k;  
            value_bucket1(t,corr_station_num1) = V1;  
            stations_opt(t,k) = 1;
        else
            stations_opt(t,k) = 0;
        end
    end
end

t_mape_error = zeros(station_num,1);
t_mae_error = zeros(station_num,1);
t_rmse_error = zeros(station_num,1);
rmse = 0;
optimal_k = [38 15 19 35 29];
t_station_m = [2 16 6 6 2];
t_station_a = [0.04 0.015 0.04 0.04 0.04];
for t = 1:station_num  
    m = t_station_m(t);
    k = optimal_k(t);
    a3 = t_station_a(t);
    mape_error =[];
    mae_error = [];
    split = 0;
    sample1 = [];
    row = 0;
    n_stations = find(station_bucket1(t,:)~=0);
    n = size(n_stations,2);
    temp1 = station_bucket1(t,:);
    n_related = temp1(n_stations);
    temp2 = value_bucket1(t,:);
    n_related_values = temp2(n_stations);
    for i=1:size(bucket4,1)
        if(i-m>0)
            row = row + 1;
            sample(row,1:m*n) = reshape(bucket4(i-m:i-1,n_ralated)',n*m,1);  %reshape
            sample(row,n*m+step) = bucket4(i,t);
            column = 0;
        end
        if(i==size(bucket4,1)-numTest)  
            split = row;  
        end    
	end
    len = size(sample1,1);
    
    X1 = sample1(:,1:m*n);
    y1 = sample1(:,m*n+1);
    
    sample_in1 = X1(1:split,:);
    sample_out1 = y1(1:split,:);
    
    input_data1 = X1(split+1:len,:);
    output_data1 = y1(split+1:len,:);
    
    param = [m n a3 k];
    
    result = D_STKNN_FUN(sample_in1,sample_out1,input_data1,param,n_related_values);  %Obtain the prediction result
    
    co_error =  abs(result - output_data1);
    mape_error = mean(co_error./output_data1,'omitnan');
    mae_error = mean(co_error,'omitnan');

    t_mape_error(t) = mean(mape_error,'omitnan');
    t_mae_error(t) = mean(mae_error,'omitnan');
    rmse = rmse + sum(co_error.^2);
    t_rmse_error(t) = sqrt(mean(co_error.^2));
    fprintf('**********************\n');
end
bucket1_mape_error = mean(t_mape_error,'omitnan')
bucket1_mae_error = mean(t_mae_error,'omitnan')
bucket1_rmse = sqrt(rmse/(size(result,1)*station_num))

