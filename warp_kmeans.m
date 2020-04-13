function [u,J,p] = warp_kmeans(X,L,s)
[K,V] = size(X); 
%% Trace Segmentations
ble(1,1) = 1;
temp1(1,1) = 0;
for i = 2:K
    temp1(i,1) = temp1(i-1,1) + norm(X(i,:)-X(i-1,:));
end
temp2 = temp1(K)/L;
for j = 1:L
    i = 1;
    while temp2*(j-1) > temp1(i,1)
        i = i+1;
        ble(j,1) = i;  
    end
end
ble(L+1,1) = K+1;
ble_ori = ble;
clear temp1,clear temp2;

for i = 1:L
    temp1{i,1} = X(ble(i,1):ble(i+1,1)-1,:);
    u(i,:) = mean(temp1{i,1},1);
    n(i,1) = length(temp1{i,1});
    temp2{i,1}=bsxfun(@minus, temp1{i,1},u(i,:));
    H(i,1)=sum(sum(temp2{i,1}.^2));
end
J_ori = sum(H); 
flag = 1;
P = 0;
J_temp = J_ori;
while(flag)
    pre_J = J_temp ;
    flag = 0;
    for j = 1:L
        if j > 1
            first = ble(j,1);
            last = first+floor(n(j,1)*(1-s)/2);
            for i = first:last
                temp1 = n(j-1,1) / (n(j-1,1)+1) * (norm(X(i,:)-u(j-1,:)))^2;
                temp2 = n(j,1) / (n(j,1)-1) * (norm(X(i,:)-u(j,:)))^2;
                d_J = temp1 - temp2;
                if n(j,1) > 1 && d_J < 0
                    flag = 1;
                    ble(j,1) = ble(j,1)+1;
                    n(j,1) = n(j,1)-1;
                    n(j-1,1) =  n(j-1,1)+1;
                    u(j,:) = u(j,:) - (X(i,:) - u(j,:))/n(j,1);
                    u(j-1,:) = u(j-1,:) + (X(i,:) - u(j-1,:))/n(j-1,1);
                    J_temp = J_temp + d_J;
                else
                    break;
                end
            end
        end
        if j < L
            last = ble(j+1,1)-1;
            first = last-floor(n(j,1)*(1-s)/2);
            for i = last:-1:first
                temp1 = n(j+1,1) / (n(j+1,1)+1) * (norm(X(i,:)-u(j+1,:)))^2;
                temp2 = n(j,1) / (n(j,1)-1) * (norm(X(i,:)-u(j,:)))^2;
                d_J = temp1 - temp2;
                if n(j,1) > 1 && d_J < 0
                    flag = 1;
                    ble(j+1,1) = ble(j+1,1) - 1;
                    n(j,1) = n(j,1) - 1;
                    n(j+1,1) = n(j+1,1) + 1;
                    u(j,:) = u(j,:) - (X(i,:) - u(j,:))/n(j,1);
                    u(j+1,:) = u(j+1,:) + (X(i,:) - u(j+1,:))/n(j+1,1);
                    J_temp = J_temp + d_J;
                else
                    break;
                end
            end
        end
    end
    P = P+1;
    J_iter(P,1) = J_temp;
end
J = [J_ori;J_iter];

for i = 1: L
    p(i,:) = [ble(i,1),ble(i+1,1)-1];
end
end
