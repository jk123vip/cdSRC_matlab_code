%function distance = cdKNN(X, y, train_label, K)
function distance = cdKNN(train_data_ori, test_data_ori, train_label, K)

%inputs:
%           train_data_ori: 总的原始训练样本矩阵
%           test_data_ori: 原始测试样本集
%           train_label: 训练样本对应的类标
%outputs:
%           distance: y对每类的K近邻距离

[m, n] = size(test_data_ori);
distance = zeros(max(train_label), n);

for p = 1:n
    y = test_data_ori(:, p);
    for i = 1:max(train_label)
        X1 = train_data_ori(:, find(train_label == i));
%         d = [];
%         for j = 1:size(X1, 2)
%             d(j) = norm(y - X1(:, j));  %d中保存y到i类每个样本点的距离
%         end
        d = sqrt(sum((repmat(y, 1, size(X1, 2)) - X1).^2));
        
        %提取前K个最小距离并取均值
        sort_d = sort(d, 'ascend'); %d按增序排列
        distance(i, p) = sum(sort_d(1:K)) / K; %K个最小距离的均值作为y对第i类的距离信息度量
    end
end

end
    