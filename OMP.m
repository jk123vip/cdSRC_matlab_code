%A-稀疏系数矩阵
%D-字典/测量矩阵（已知）
%X-测量值矩阵（已知）
%L-稀疏层

function A=OMP(D,X,L)
[n,P]=size(X);
[n,K]=size(D);
for k=1:1:P
    a=[];
    x=X(:,k);
    residual=x;%残差
    %r = zeros(1, P);
    indx=zeros(L,1);%索引集
    for j=1:1:L
        proj=D'*residual;%D转置与residual相乘，得到与residual与D每一列的内积值
        pos=find(abs(proj)==max(abs(proj)));%找到内积最大值的位置
        pos=pos(1);%若最大值不止一个，取第一个
        indx(j)=pos;%将这个位置存入索引集的第j个值
        a=pinv(D(:,indx(1:j)))*x;%indx(1:j)表示第一列前j个元素
        residual=x-D(:,indx(1:j))*a;
    end
    %r(k) = norm(residual);
    temp=zeros(K,1);
    temp(indx)=a;
    A(:,k)=temp;%只显示非零值及其位置
end
return