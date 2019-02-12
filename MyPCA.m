function [ conf_mats ] = MyPCA( data_set )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Centre and Standardize
A = zscore(data_set(:,1:end-1));

%diagonal matrix D of eigenvalues and 
%matrix V whose columns are the corresponding right eigenvectors,
[V,D] = eig(cov(A));
eigen_values = flipud(diag(D));
eigen_vectros = fliplr(V);
n_features = size(eigen_values,1);
conf_mats = zeros(2,2,size(n_features,1));

for m = 1:n_features
    choosed_eigens_V = eigen_vectros(:,1:m);
    new_features = A * choosed_eigens_V;
    new_data_set = [new_features  data_set(:,end)];
    
    [trainInd,valInd,~] = dividerand(new_data_set',0.5,0.5,0);
    conf_mat = MyKNN(trainInd',valInd',5);
    conf_mats(:,:,m) = conf_mat;
end
    


end

