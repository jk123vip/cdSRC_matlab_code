function [train_index, test_index] = select_train_data(labels, percent)
%输入：
%           labels: 所有数据对应的类标
%           percent: 每类选取的训练样本比例
%输出：
%           train_index: 选取的训练数据的下标
%           test_index: 剩下的作为测试数据下标

num_of_classes = max(labels);
train_index = [];
test_index = [];
for i = 1:num_of_classes
    
    %随机选取一定比例的训练样本
    ind = find(labels == i);
    L = length(ind);
    p = randperm(L);
    num = round(L*percent);
    train_index = [train_index, ind(p(1:num))];
    test_index = [test_index, ind(p(num+1:end))];
    
%     %选取靠前的一定比例的训练样本
%     ind = find(labels == i);
%     L = length(ind);
%     num = round(L * percent);
%     train_index = [train_index, ind(1:num)];
%     test_index = [test_index, ind(num+1:end)];
end

end