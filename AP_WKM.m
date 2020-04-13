clc
clear all
close

load dataMatrix.mat %Read data

x = zscore(dataMatrix);
fprintf('Affinity Propagation (APCLUSTER) sample/demo code\n\n');
N=size(x,1); 
M=N*N-N; s=zeros(M,3); %Make ALL N^2-N similarities\n');
j=1;
for i=1:N
    for k=[1:i-1,i+1:N]
        s(j,1)=i; s(j,2)=k; s(j,3)=-sum((x(i,:)-x(k,:)).^2);
        j=j+1;
    end
end
p=median(s(:,3)); %Set preference to median similarity\n');
[idx,netsim,dpsim,expref]=AP(s,p,'plot');   %Call AP algorithm to identify similar traffic patterns for road segments; 
fprintf('Number of clusters: %d\n',length(unique(idx)));
fprintf('Fitness (net similarity): %g\n',netsim);

i = unique(idx); 
optimal_k = zeros(5,1);
for j=1:size(i)
    c = 1e-5;  %Square error sum (iteration stop condition)
    s = 0.2;   %Adjustment factor,0 <= s <=1
    temp_L = zeros(5,10);
    for L=2:20       
        [u_wkm,J_wkm,p_wkm] = warp_kmeans(x(i(j),:)',L,s);  %Call warp-kmeans algorithm to divide time period£»
        temp = zeros(288,1);
        index_c = 1;
        for kk=1:L
            temp(p_wkm(kk,1):p_wkm(kk,2),1)=index_c;
            index_c = index_c + 1;
        end
        temp_L(j,L) = mean(silhouette(x(i(j),:)',temp));    %Compute silhouette;
    end
    [aa,bb] = max(temp_L(j,:));
    optimal_k(j) = bb;
    fprintf('The %d silhouette: %g\n',j,aa);
end

i = unique(idx); 
for j=1:size(i) %Walk through each traffic pattern;
    c = 1e-5; 
    s = 0.2;
    [u_wkm,J_wkm,p_wkm] = warp_kmeans(x(i(j),:)',optimal_k(j),s); 
    figure;     %Draw the figure;
    ii=find(idx==i(j));
    tt=1:288;
    col=rand(1,3);
    plot(tt,x(ii,:),'Color',col);
    yy=get(gca,'ylim');
    hold on;
    plot(tt,x(i(j),:),'b','LineWidth',1);
    for hh=1:optimal_k(j)
        plot([p_wkm(hh,2) p_wkm(hh,2)],yy,'--r','LineWidth',2)
    end
end



