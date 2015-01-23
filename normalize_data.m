function data = normalize_data(data)
%L2范数行归一化

[m, n] = size(data);
for i = 1:n
    data(:, i) = data(:, i) / norm(data(:, i));
end

end
