close all;
clear;
clc;
load('Indiana200.mat');
Indiana_data=fea';
Indiana_labels=labels';
% load('Indiana_200.mat');
% data_ori = Indiana_data(find(Indiana_labels ~= 0), :)';
% labels = Indiana_labels(find(Indiana_labels ~= 0), :)';

lyl=[2,3,5,6,8,10,11,12,14];
data_ori =[];
labels=[];
for j=1:9
data_ori =[data_ori,Indiana_data(find(Indiana_labels == lyl(j)), :)' ];
labels = [labels ,repmat(j,1,length(Indiana_labels(find(Indiana_labels == lyl(j)), :)'))];
end
%数据归一化
data = normalize_data(data_ori);

%对原始数据进行LFDA映射降到30维，用于后面的cdKNN
[T,ZZ]=lda(data_ori', labels', 30);   %T是映射矩阵，Z是降维后的数据T'*X
Z=ZZ';
% [T,Z]=LFDA(data_ori, labels, 30);   %T是映射矩阵，Z是降维后的数据T'*X

%选取训练样本和测试样本
%select_train_data.m用来按比例选择训练样本
%select_train_data1.m用来按个数选择训练样本
percent = 0.1; %每类样本中训练样本比例
%N = 10;    %每类取N个作为训练样本
[train_index, test_index] = select_train_data(labels, 0.1);
% [train_index, test_index] = select_train_data1(labels, 30);

%用于cdOMP的归一化后数据
train_data = data(:, train_index);
train_label = labels(train_index);
test_data = data(:, test_index);
test_label = labels(test_index);

%用于cdKNN的降维数据
train_data_ori = Z(:, train_index);
test_data_ori = Z(:, test_index);

X = train_data;
S = 10;
K = 10;
lambda = 0.0001;
c = max(labels);

%cdOMP迭代程序，得到【类别数*测试样本个数】大小的矩阵residual
tic
residual = zeros(max(train_label), length(test_label));
distance = zeros(size(residual));
for i = 1:max(train_label)
    X1 = X(:, find(train_label == i));
    A= OMP(X1, test_data, S);
    nor = sqrt(sum((X1 * A - test_data).^2));
    residual(i, :) = nor;
end

%cdKNN程序，得到【类别数*测试样本个数】大小的矩阵distance
distance = cdKNN(train_data_ori, test_data_ori, train_label, K);

%整合相关度信息和欧氏距离信息
w = residual + lambda * distance;

%计算正确率
result = zeros(1, length(test_label));
for i = 1:length(test_label)
    result(i) = find(w(:, i) == min(w(:, i)));
end
per = sum(result == test_label)/length(test_label)
toc;




% tic;
% %result = zeros(1, 1000);
% w = zeros(1, length(test_label));   %w用来保存新的类标
% for i = 1:length(test_label)
%     residual = cdOMP(X, test_data(:, i), train_label, S);   %差值，y关于每类的相关性度量
%     distance = cdKNN(X, test_data(:, i), train_label, K);  %距离，y关于每类的欧氏距离性度量
%     W = residual + lambda * distance;
%     w(i) = find(W == min(W));
% end
% %a = test_label(1:1000);
% per = length(find(test_label == w))
% %percent = lenght(find(w == test_label(1:100)))/100;
% toc;