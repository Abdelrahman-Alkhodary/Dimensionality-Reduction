function [ conf_mat ] = MyKNN( train_data,val_data,k )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
conf_mat = zeros(2,2);
%conf_mat rows represents the predicted value
%conf_mat cols represents the actual value


[rows_tr, cols_tr] = size(train_data);
[rows_val, cols_val] = size(val_data);

euclideanDistance = zeros(rows_tr,2);

for i_val =1:rows_val
    actual = val_data(i_val,end);
    actual;
    for i_row_tr =1:rows_tr
        distance = 0; 
        for i_col =1:cols_tr-1
            distance = distance + (val_data(i_val,i_col) - train_data(i_row_tr,i_col))^2;
        end
        %distance to test instance
        euclideanDistance(i_row_tr,1) = sqrt(distance);
        %class of training
        euclideanDistance(i_row_tr,2) = train_data(i_row_tr,end);
    end
    vote_mat = inf(1,5);
    for i=1:k
        %get the index of the instance with the min distance to the test
        %point
        [~,index] = min(euclideanDistance(:,1));
        %replace it with infinity
        euclideanDistance(index,1) = inf;
        %get the class of the training instance with the min distance
        vote_mat(i) = euclideanDistance(index,2);
    end
    one_vote = sum(vote_mat==1);
    zero_vote = sum(vote_mat==0);
    if one_vote > zero_vote
        predicted = 1;
    else
        predicted = 0;
    end
    predicted ;
    conf_mat(predicted+1, actual+1) = conf_mat(predicted+1, actual+1) +1;
end