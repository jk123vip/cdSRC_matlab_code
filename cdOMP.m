function residual = cdOMP(X, y, train_label, S)


residual = zeros(max(train_label), 1);
for i = 1:max(train_label)
    X1 = X(:, find(train_label == i));
    LAMBDA = [];
    r = y;
    for m = 1:S
        nei_ji = r' * X1;            %内积
        lambda = find(nei_ji == max(nei_ji));
        LAMBDA = [LAMBDA, lambda];
        X2 = X1(:, LAMBDA);             %X2: 第i类训练样本中参加表示的样本集合
        alpha = zeros(size(X2, 2), 1);                %alpha: 用X2重构y时的参数向量
        
        %梯度下降法求alpha和r
        a = 0.5;
        num_iters = 100;
        [alpha, r] = gradient_descent(X2, y, alpha, a, num_iters);               %a是alpha迭代公式中的参数，num_iters是最大迭代次数
    end
    
    residual(i) = norm(r);
end

end