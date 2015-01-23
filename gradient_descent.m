function [alpha, r] = gradient_descent(X2, y, alpha, a, num_iters)
%inputs:
%           X1: 训练样本
%           y: 测试样本
%           alpha: 参数向量
%           a: 
%           num_iters: 最大迭代次数
%outputs:
%           alpha: 参数
%           r: 差值向量

m = length(y);
%A = [];
for i = 1:num_iters
    alpha = alpha - a / m * X2' * (X2 * alpha - y);
    r = y - X2 * alpha;
    %A(i) = norm(r);
end
%plot(A);

end