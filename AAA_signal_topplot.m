%% signal final
% Dianwen Jin
% dj2484
clc;
% clear; 
close all; 
load Subject_2;
% topoplot([],'standard60.loc','style','map');

%%
% sub1&2&4&7=375-450,sub3=525-600;sub5=365-440
T1 = 350;
T2 = 425;
testpre = X_EEG_TRAIN(:,T1:T2,:);
X = reshape(testpre,size(testpre,2)*size(testpre,1),size(testpre,3));
row = size(X,1);
X(row+1,:) = Y_EEG_TRAIN(:);
colunmrank = randperm(size(X,2));
XY = X(:,colunmrank);
Xtrain = XY(:,1:55);
Xtest = XY(1:end-1,56:end);

yfit = trainedModel8.predictFcn(Xtest);
table(yfit,XY(size(XY,1),56:end)','VariableNames',{'PredictedLabel','TrueLabel'})
sum = 0;
for i = 1:size(XY,2)-55
    if yfit(i) == XY(size(XY,1),55+i)
        sum = sum+1;
    end
end
accuracy = sum/(size(XY,2)-55);

pre = X_EEG_TEST(:,T1:T2,:);
Xpre = reshape(pre,size(pre,2)*size(pre,1),size(pre,3));
ytest = trainedModel8.predictFcn(Xpre);


%% 脸车的大脑图
% % seperate the training data into car and face
% num_trails = size(Y_EEG_TRAIN,1);
% m = 1; n = 1;
% for i = 1:num_trails
%     if Y_EEG_TRAIN(i,1) == 1
%         Face(:,:,m) = X_EEG_TRAIN(:,:,i);
%         Y_face(m,1) = Y_EEG_TRAIN(i,1);
%         m =m+1;
%     else
%         Car(:,:,n) = X_EEG_TRAIN(:,:,i);
%         Y_car(n,1) = Y_EEG_TRAIN(i,1);
%         n = n+1;
%     end
% end
% 
% % get the mean of each trail, result in:60*1200*1
% data1 = mean(Face,3);
% data2 = mean(Car,3);
% % % 试试单个通道差异
% % data1 = Face(:,:,5);
% % data2 = Car(:,:,5);
% % squeeze the channel that =1, 3d->2d: 60*1200
% Face2d = squeeze(data1);
% Car2d = squeeze(data2);
% % 选取看到图片后0-200ms取平均值，压成1维
% Face1dpre = Face2d(:,250:350);
% Car1dpre = Car2d(:,250:350);
% mean1 = mean(Face1dpre,2);
% mean2 = mean(Car1dpre,2);
% Face1d = squeeze(mean1);
% Car1d = squeeze(mean2);
% % 选前150秒当做啥也没看见的图
% data3 = mean(X_EEG_TRAIN,3);
% Whole2d = squeeze(data3);
% no1dpre = Whole2d(:,1:150);
% mean3 = mean(no1dpre,2);
% no1d = squeeze(mean3);
% 
% figure
% subplot(2,2,1)
% topoplot(Face1d,'standard60.loc','style','map');
% title("Face");
% subplot(2,2,2)
% topoplot(Car1d,'standard60.loc','style','map');
% title("Car");
% subplot(2,2,3)
% topoplot(no1d,'standard60.loc','style','map');
% title("Nothing");
% 
% eegplot(X_EEG_TRAIN);
% %eegplot(Whole2d);

%% for循环切分数据
% timecount = X_EEG_TRAIN(:,151:450,11:60);
% Bigtrain = zeros(50,18001);
% for i = 1:50
%     for j = 1:60
%     Bigtrain(i,(j-1)*300+1:j*300) = timecount(j,:,i);
%     end
%     Bigtrain(i,18001) = Y_EEG_TRAIN(i+10);
% end
% 
% fortest1 = X_EEG_TRAIN(:,151:450,1:10);
% fortest2 = X_EEG_TRAIN(:,151:450,61:74);
% Bigtest = zeros(24,18001);
% for i = 1:10
%     for j = 1:60
%     Bigtest(i,(j-1)*300+1:j*300) = fortest1(j,:,i);
%     end
%     Bigtest(i,18001) = Y_EEG_TRAIN(i);
% end
% for i = 1:14
%     for j = 1:60
%     Bigtest(i+10,(j-1)*300+1:j*300) = fortest2(j,:,i);
%     end
%     Bigtest(i+10,18001) = Y_EEG_TRAIN(i+60);
% end

% %%
% X = X_EEG_TRAIN(:,151:500,1:50);
% XY = reshape(X,21000,50);
% XY(21000+1,:) = Y_EEG_TRAIN(1:50);
% XY = XY';
% 
% X2 = X_EEG_TRAIN(:,151:450,11:60);
% XY2 = reshape(X2,18000,50);
% XY2(18000+1,:) = Y_EEG_TRAIN(11:60);
% XY2 = XY2';
% 
% 
% testing = X_EEG_TRAIN(:,151:450,51:74);
% testY = reshape(testing,18000,24);
% testY(18000+1,:) = Y_EEG_TRAIN(51:74);
% testY =testY';
% 
% yfit = trainedModel1.predictFcn(XY);
% [label,score] = predict(trainedModel_CoarseSVM,testY);
% [label,score] = predict(trainedModel1,testY);
% table(Y_EEG_TRAIN(51:74),label(1:24),score(1:24,1),'VariableNames',{'TrueLabel','PredictedLabel','Score'})

%% SVM
% testpre = X_EEG_TRAIN(:,375:450,:);
% X = mean(testpre,2);% Average time
% X = squeeze(X);
% X(60+1,:) = Y_EEG_TRAIN(:);
% colunmrank = randperm(size(X,2));
% XY = X(:,colunmrank);
% Xtrain = XY(:,1:54);
% % Xtrain(60+1,:) = Y_EEG_TRAIN(1:54);
% 
% 
% Xtest = XY(1:60,55:end);
% % Xtest(60+1,:) = Y_EEG_TRAIN(51:74);
% % 
% yfit = trainedModel.predictFcn(Xtest);
% % test = X(:,51:74);
% % [label,score] = predict(trainedModel2,test);
% table(yfit,XY(61,55:end)','VariableNames',{'PredictedLabel','TrueLabel'})
% sum = 0;
% for i = 1:size(XY,2)-54
%     if yfit(i) == XY(61,54+i)
%         sum = sum+1;
%     end
% end
% accuracy = sum/(size(XY,2)-54);

%% 测试reshape
% A = ones(3,4,5);
% n = 0;
% for i = 1:3
%     for j = 1:4
%         for k = 1:5
%         A(i,j,k) = n;
%         n =n+1;
%         end
%     end
% end

%% SVM with poly2 kernel
% % get the mean of each channel, result in:60*1*74
% data1 = mean(X_EEG_TRAIN,2);
% data2 = mean(X_EEG_TEST,2);
% % squeeze the channel that =1, 3d->2d: 64*79
% train2d = squeeze(data1);
% test2d = squeeze(data2);
% 
% % training 
% tic;
% Ytrain = Y_EEG_TRAIN';
% SVMmodel = fitcsvm(train2d,Ytrain);
% % SVMmodel = svmtrain(train2d, Y_EEG_TRAIN);
% t1 = toc;
% % classification
% tic;
% [label,score] = predict(SVMmodel,test2d);
% t2 = toc;
% disp(num2str(t1));
% disp(num2str(t2));

%% Yimin
% svmChannel = squeeze(mean(X_EEG_TRAIN,2))';
% SVMModel = fitcsvm(svmChannel,Y_EEG_TRAIN','Holdout',0.15,'ClassNames',{'1','0'},'Standardize',true);
% CompactSVMModel = SVMModel.Trained{1}; % Extract trained, compact classifier
% testInds = test(SVMModel.Partition);   % Extract the test indices
% XTest = squeeze(mean(X_EEG_TEST,2))';
% % XTest = squeeze(mean(X_EEG_TRAIN(:,:,5:10),2))';
% % YTest = Y(testInds,:);
% [label,score] = predict(CompactSVMModel,XTest);
