%%
% Dianwen Jin
% dj2484
%时间50取平均得到60*24*74，reshape到1440*74，不pca直接求，0,0.55
clc;
clear; 
close all; 
load Subject_1;
%%
SmallerXtrain = zeros(60,24,74);
for i = 1:24
    X = X_EEG_TRAIN(:,(i-1)*50+1:i*50,:);
    meantime = mean(X,2);
    SmallerXtrain(:,i,:) = meantime;
end

num_trails = size(Y_EEG_TRAIN,1);
m = 1; n = 1;
for i = 1:num_trails
    if Y_EEG_TRAIN(i,1) == 1
        Face(:,:,m) = SmallerXtrain(:,:,i);
        Y_face(m,1) = Y_EEG_TRAIN(i,1);
        m =m+1;
    else
        Car(:,:,n) = SmallerXtrain(:,:,i);
        Y_car(n,1) = Y_EEG_TRAIN(i,1);
        n = n+1;
    end
end

% 取平均后reshape，0.9,0.3
data1 = reshape(Face,1440,size(Face,3));
data2 = reshape(Car,1440,size(Car,3));
% get the mean of each test
mean1 = mean(data1,2);
mean2 = mean(data2,2);
% cov of 
cov1 = cov(data1');
cov2 = cov(data2');
aver_cov = (cov1 + cov2)./2;
% Class Probabilities
num1 = size(Face,3);
num2 = size(Car,3);
pc1 = num1/(num2+num1);    
pc2 = num2/(num2+num1); 
% separation vector & bias
v = pinv(aver_cov)*(mean1-mean2); 
v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
% get the l and p(c|y)
scores_1 = zeros(num1,1);
for k = 1:num1
    l = v'*data1(:,k) + v0;
    scores_1(k) = 1/(1+exp(-l));
end
scores_2 = zeros(num1,1);
for k = 1:num2
    l = v'*data2(:,k) + v0;
    scores_2(k) = 1/(1+exp(-l));
end
scores = [scores_1;scores_2];
% figure
% plot(scores_1);
% hold on 
% plot(scores_2);

%% v and ROC
figure(1)
stem(v);
hold on
plot(v);
title('v');
% set up a standard, 长度为num1+num2, 将算出来的score和它对应
labels = [ones(1,num1), zeros(1,num2)];
[X,Y,~,Az_AllTrainingData] = perfcurve(labels,scores,1);
figure(2)
plot(X,Y);
title('ROC - All Training Data');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_All_Training_Data:',num2str(Az_AllTrainingData)])

%% Classification - Leave-One-Out Set
scores = zeros(1,num1+num2);
data1All = data1;
data2All = data2;

% For each sample in class 1 & 2, compute p(c|y)
for k = 1:num1
    % Leave 1 out
    dataOneOut = data1All(:,k);
    data1 = data1All;
    data1(:,k) = [];  %清空，后面的数补上来
    mean1 = mean(data1,2);
    mean2 = mean(data2,2);
    cov1 = cov(data1');
    cov2 = cov(data2');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num1-1)/(num2+num1-1);   
    pc2 = num2/(num2+num1-1);       
    v = pinv(aver_cov)*(mean1-mean2);
    v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(k) = 1/(1+exp(-l)); 
end

for k = 1:num2
    dataOneOut = data2All(:,k);
    data2 = data2All;
    data2(:,k) = [];
    mean1 = mean(data1,2);
    mean2 = mean(data2,2);
    cov1 = cov(data1');
    cov2 = cov(data2');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num1)/(num2+num1-1);       
    pc2 = (num2-1)/(num2+num1-1);      
    v = pinv(aver_cov)*(mean1-mean2);
    v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(num1+k) = 1/(1+exp(-l));
end

[X,Y,T,Az_LeaveOneOut] = perfcurve(labels,scores,1);
figure(3)
plot(X,Y);
title('ROC - Leave One Out');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_Leave_One_Out:',num2str(Az_LeaveOneOut)])

