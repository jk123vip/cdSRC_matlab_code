function [train_index, test_index] = select_train_data1(labels, N)
%   labels: data labels
%   N: the number of training examples each class

c = max(labels);
train_index = [];
test_index = [];

for i = 1:c
    ind = find(labels == i);
    if N < length(ind)
        L = length(ind);
        p = randperm(L);
        train_index = [train_index, ind(p(1:N))];
        test_index = [test_index, ind(p(N+1:end))];
    else
       train_index = [train_index, ind];
       test_index = [test_index, ind];
    end
end

end