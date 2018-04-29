%%
% Dianwen Jin
% dj2484
% 275-350，没有pca，结果1,0.57
clc;
clear; 
close all; 
load Subject_4;

%% seperate the training data into car and face
num_trails = size(Y_EEG_TRAIN,1);
m = 1; n = 1;
for i = 1:num_trails
    if Y_EEG_TRAIN(i,1) == 1
        Face(:,:,m) = X_EEG_TRAIN(:,:,i);
        Y_face(m,1) = Y_EEG_TRAIN(i,1);
        m =m+1;
    else
        Car(:,:,n) = X_EEG_TRAIN(:,:,i);
        Y_car(n,1) = Y_EEG_TRAIN(i,1);
        n = n+1;
    end
end

%% linear discriminator
% get the meanful time
T1 = 325;T2=375;
pre1 = Face(:,T1:T2,:);
pre2 = Car(:,T1:T2,:);
% get the mean of time, result in:60*1*37
data1 = mean(pre1,2);
data2 = mean(pre2,2);
% squeeze the channel that =1, 3d->2d: 60*37
data1 = squeeze(data1);
data2 = squeeze(data2);

% get the mean of each test
mean1 = mean(data1,2);
mean2 = mean(data2,2);
% cov of 
cov1 = cov(data1');
cov2 = cov(data2');
aver_cov = (cov1 + cov2)./2;
% Class Probabilities
num1 = size(pre1,3);
num2 = size(pre2,3);
pc1 = num1/(num2+num1);    
pc2 = num2/(num2+num1); 
% separation vector & bias
v = aver_cov\(mean1-mean2); 
v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
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

% Xtraining = X_EEG_TRAIN (:,T1:T2,:);
% Xtraining = mean(Xtraining,2);
% Xtraining = squeeze(Xtraining);
% Xtest = X_EEG_TEST (:,T1:T2,:);
% Xtest = mean(Xtest,2);
% Xtest = squeeze(Xtest);
% % row：observation，column：feature
% % sample&training 相同的列数，group&training相同的行数
% [class,err] = classify(Xtest',Xtraining',scores);
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
    v = aver_cov\(mean1-mean2);
    v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
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
    v = aver_cov\(mean1-mean2);
    v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
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
