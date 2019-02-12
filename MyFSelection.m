function [ best_conf_mat ] = MyFSelection( data_set )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

choosed_data_set = [];
[~, cols] = size(data_set);
num_features = cols-1;
best_conf_mat = zeros(2,2,num_features);
for L = 1:num_features
    
    acc_index = 1;
    for i = L:num_features
        %dataset with best accuracies features
        new_data_set = [choosed_data_set data_set(:,i) data_set(:,end)];
        [trainInd,valInd,~] = dividerand(new_data_set',0.5,0.5,0);
        val_data = valInd';
        train_data = trainInd';
        accuracy_vector = zeros(2,num_features-L+1);
        conf_mat_all = zeros(2,2,num_features-L+1);
        conf_mat = MyKNN( train_data,val_data,5 );
        conf_mat_all(:,:,acc_index) = conf_mat;
        %store the accuracy of the selected feature with its index in the
        %original dataset
        accuracy_vector(1, acc_index) = (conf_mat(1,1) + conf_mat(2,2))/sum(sum(conf_mat)); 
        accuracy_vector(2, acc_index) = i;
        acc_index = acc_index+1;
    end
    %get the index of the best accuarcy in the vector so we can get the
    %index of the coressponding feature
    [~,index] = max(accuracy_vector(1,:));
    feature_index = accuracy_vector(2,index);
    %store the feature with highest accuracy
    choosed_data_set =[choosed_data_set data_set(:,feature_index)];
    %store the conf matrix with highest accuracy
    best_conf_mat(:,:,L) = conf_mat_all(:,:,index);
    %swap the features 
    temp = data_set(:,L);
    data_set(:,L) = data_set(:,feature_index);
    data_set(:,feature_index) = temp;
        



end


end

