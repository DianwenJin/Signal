%%
% Dianwen Jin
% dj2484
% 275-350取平均后求pca，结果0,0.57
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

if m>n
    Face = Face(:,:,1:n-1);
    elseif n>m
    Car = Car(:,:,1:m-1);
end


%% linear discriminator
% 取部分reshape，结果0.5,0.37
pre1 = Face(:,325:375,:);
pre2 = Car(:,325:375,:);
% 3060*36, feature*trail
data_face = reshape(pre1,size(pre1,2)*size(pre1,1),size(pre1,3));
data_car = reshape(pre2,size(pre2,2)*size(pre2,1),size(pre2,3));
% % % 转置，37*4500，行37表示一个样本，列4500表示特征变量
% pca_face = pca_face';
% pca_car = pca_car';

% get the mean of each trail
mean1 = mean(data_face,2);
mean2 = mean(data_car,2);
% cov of 
cov1 = cov(data_face');
cov2 = cov(data_car');
aver_cov = (cov1 + cov2)./2;

% Class Probabilities, size of trail
num_face_trail = size(data_face,2);
num_car_trail = size(data_car,2);
pc1 = num_face_trail/(num_car_trail+num_face_trail);    
pc2 = num_car_trail/(num_car_trail+num_face_trail); 
% separation vector & bias
v =pinv(aver_cov)*(mean1-mean2); 
% v =aver_cov\(mean1-mean2);
v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
%v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
% get the l and p(c|y)
scores_1 = zeros(num_face_trail,1);
for k = 1:num_face_trail
    l = v'*data_face(:,k) + v0;
    scores_1(k) = 1/(1+exp(-l));
end
scores_2 = zeros(num_car_trail,1);
for k = 1:num_car_trail
    l = v'*data_car(:,k) + v0;
    scores_2(k) = 1/(1+exp(-l));
end
scores = [scores_1;scores_2];
% figure(4)
% plot(scores_1);
% hold on 
% plot(scores_2);

%% v and ROC
% figure(1)
% stem(v);
% hold on
% plot(v);
% title('v');
% set up a standard, 长度为num1+num2, 将算出来的score和它对应
labels = [ones(1,num_face_trail), zeros(1,num_car_trail)];
[X,Y,~,Az_AllTrainingData] = perfcurve(labels,scores,1);
figure(2)
plot(X,Y);
title('ROC - All Training Data');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_All_Training_Data:',num2str(Az_AllTrainingData)])

% Classification - Leave-One-Out Set
scores = zeros(1,num_face_trail+num_car_trail);
data1All = data_face;
data2All = data_car;

% For each sample in class 1 & 2, compute p(c|y)
for k = 1:num_face_trail
    % Leave 1 out
    dataOneOut = data1All(:,k);
    data_face = data1All;
    data_face(:,k) = [];  %清空，后面的数补上来
    mean1 = mean(data_face,2);
    mean2 = mean(data_car,2);
    cov1 = cov(data_face');
    cov2 = cov(data_car');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num_face_trail-1)/(num_car_trail+num_face_trail-1);   
    pc2 = num_car_trail/(num_car_trail+num_face_trail-1); 
%     v = aver_cov\(mean1-mean2);
     v = pinv(aver_cov)*(mean1-mean2);
     v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
%     v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(k) = 1/(1+exp(-l)); 
end

for k = 1:num_car_trail
    dataOneOut = data2All(:,k);
    data_car = data2All;
    data_car(:,k) = [];
    mean1 = mean(data_face,2);
    mean2 = mean(data_car,2);
    cov1 = cov(data_face');
    cov2 = cov(data_car');
    aver_cov = (cov1 + cov2)./2;
    pc1 = (num_face_trail)/(num_car_trail+num_face_trail-1);       
    pc2 = (num_car_trail-1)/(num_car_trail+num_face_trail-1);  
%    v = aver_cov\(mean1-mean2);
    v = pinv(aver_cov)*(mean1-mean2);
     v0 = -0.5*mean1'*(pinv(aver_cov)*mean1) + 0.5*mean2'*(pinv(aver_cov)*mean2) + log(pc1/pc2);
%    v0 = -0.5*mean1'*(aver_cov\mean1) + 0.5*mean2'*(aver_cov\mean2) + log(pc1/pc2);
    l = v'*dataOneOut + v0;
    scores(num_face_trail+k) = 1/(1+exp(-l));
end

[X,Y,T,Az_LeaveOneOut] = perfcurve(labels,scores,1);
figure(3)
plot(X,Y);
title('ROC - Leave One Out');
xlabel('false positive rate');
ylabel('true positive rate');
disp(['Az_Leave_One_Out:',num2str(Az_LeaveOneOut)])
