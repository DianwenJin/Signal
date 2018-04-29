%%
% Dianwen Jin
% dj2484
% 275-350取平均后求pca，结果0,0.57
clc;
% clear; 
close all; 
load Subject_1;

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

if m>n
    Face = Face(:,:,1:n-1);
elseif n>m
    Car = Car(:,:,1:m-1);
end

%% linear discriminator
% 取部分reshape，结果0.5,0.37
pre1 = Face(:,325:375,:);
pre2 = Car(:,325:375,:);
pca_face = reshape(pre1,size(pre1,2)*size(pre1,1),size(pre1,3));
pca_car = reshape(pre2,size(pre2,2)*size(pre2,1),size(pre2,3));
% 转置，37*4500，行37表示一个样本，列4500表示特征变量
pca_face = pca_face';
pca_car = pca_car';

% PCA主成分分析
[coeff1,score1,latent1] = pca(pca_face);
[coeff2,score2,latent2] = pca(pca_car);
percentage1 = cumsum(latent1)./sum(latent1);
percentage2 = cumsum(latent2)./sum(latent2);
index1 = find(percentage1>=0.95);
num_eigen_face = index1(1);
index2 = find(percentage2>=0.95);
num_eigen_car = index2(1);
pca_face = score1(:,1:num_eigen_face);
pca_car = score2(:,1:num_eigen_car);

% get the mean of each trail
mean1 = mean(pca_face,2);
mean2 = mean(pca_car,2);
% cov of 
cov1 = cov(pca_face');
cov2 = cov(pca_car');
aver_cov = (cov1 + cov2)./2;

% Class Probabilities
% num_eigen_face = size(pca_face,2);
% num_eigen_car = size(pca_car,2);
pc1 = num_eigen_face/(num_eigen_car+num_eigen_face);    
pc2 = num_eigen_car/(num_eigen_car+num_eigen_face); 
% separation vector & bias
v =pinv(aver_cov)*(mean1-mean2); 
% v =aver_cov\(mean1-mean2);
v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
% get the l and p(c|y)
scores_1 = zeros(num_eigen_face,1);
for k = 1:num_eigen_face
    l = v'*pca_face(:,k) + v0;
    scores_1(k) = 1/(1+exp(-l));
end
scores_2 = zeros(num_eigen_car,1);
for k = 1:num_eigen_car
    l = v'*pca_car(:,k) + v0;
    scores_2(k) = 1/(1+exp(-l));
end
scores = [scores_1;scores_2];
% figure(4)
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
labels = [ones(1,num_eigen_face), zeros(1,num_eigen_car)];
[X,Y,~,Az_AllTrainingData] = perfcurve(labels,scores,1);
figure(2)
plot(X,Y);
title('ROC - All Training Data');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_All_Training_Data:',num2str(Az_AllTrainingData)])

% Classification - Leave-One-Out Set
scores = zeros(1,num_eigen_face+num_eigen_car);
data1All = pca_face;
data2All = pca_car;

% For each sample in class 1 & 2, compute p(c|y)
for k = 1:num_eigen_face
    % Leave 1 out
    dataOneOut = data1All(:,k);
    pca_face = data1All;
    pca_face(:,k) = [];  %清空，后面的数补上来
    mean1 = mean(pca_face,2);
    mean2 = mean(pca_car,2);
    cov1 = cov(pca_face');
    cov2 = cov(pca_car');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num_eigen_face-1)/(num_eigen_car+num_eigen_face-1);   
    pc2 = num_eigen_car/(num_eigen_car+num_eigen_face-1);       
    v = pinv(aver_cov)*(mean1-mean2);
    v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(k) = 1/(1+exp(-l)); 
end

for k = 1:num_eigen_car
    dataOneOut = data2All(:,k);
    pca_car = data2All;
    pca_car(:,k) = [];
    mean1 = mean(pca_face,2);
    mean2 = mean(pca_car,2);
    cov1 = cov(pca_face');
    cov2 = cov(pca_car');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num_eigen_face)/(num_eigen_car+num_eigen_face-1);       
    pc2 = (num_eigen_car-1)/(num_eigen_car+num_eigen_face-1);      
    v = pinv(aver_cov)*(mean1-mean2);
    v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(num_eigen_face+k) = 1/(1+exp(-l));
end

[X,Y,T,Az_LeaveOneOut] = perfcurve(labels,scores,1);
figure(3)
plot(X,Y);
title('ROC - Leave One Out');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_Leave_One_Out:',num2str(Az_LeaveOneOut)])
