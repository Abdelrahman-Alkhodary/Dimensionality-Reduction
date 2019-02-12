clc;
clear;
%fprintf("In the file import_dataset enter the dataset.txt destination \n Press any key to continue \n");pause;
import_dataset;
data_set = table2array(dataset);
[trainInd,valInd,~] = dividerand(data_set',0.5,0.5,0);
val_data = valInd';
train_data = trainInd';
k = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%     KNN     %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 [ conf_mat_KNN ] = MyKNN( train_data,val_data,k );
 conf_mat_KNN; 

accuracy_KNN = ( conf_mat_KNN(1,1) + conf_mat_KNN(2,2) ) / sum(sum(conf_mat_KNN));
precision_KNN = conf_mat_KNN(1,1) / (conf_mat_KNN(1,1) + conf_mat_KNN(2,1));
recall_KNN = conf_mat_KNN(1,1) / (conf_mat_KNN(1,1) + conf_mat_KNN(1,2));

l_knn=[accuracy_KNN precision_KNN recall_KNN];
stem(l_knn)
title('accuracy precision recall KNN')

fprintf(" The Accuracy for KNN Model = %f \n" ,accuracy_KNN)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%     PCA     %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_set_pca = data_set;
 [ conf_mats_PCA ] = MyPCA( data_set_pca );
 conf_mats_PCA;

accuracy_PCA = zeros(1,57);
precision_PCA = zeros(1,57);
recall_PCA = zeros(1,57);

for i = 1:57
    conf_mat = conf_mats_PCA(:,:,i);
    accuracy_PCA(i) = ( conf_mat(1,1) + conf_mat(2,2) ) / sum(sum(conf_mat));
    precision_PCA(i) = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
    recall_PCA(i) = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
end
m =1:57;
figure
plot(m,accuracy_PCA)
title('Accuracy for PCA')

figure
plot(m,precision_PCA)
title('Precision for PCA')

figure
plot(m,recall_PCA)
title('Recall for PCA')


conf_mat = conf_mats_PCA(:,:,2);
accuracy_PCA_2 = ( conf_mat(1,1) + conf_mat(2,2) ) / sum(sum(conf_mat));
precision_PCA_2 = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
recall_PCA_2 = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
    
l_pca=[accuracy_PCA_2 precision_PCA_2 recall_PCA_2];
figure
stem(l_pca)
title('accuracy precision recall PCA')

[acc_pca,i_pca]=max(accuracy_PCA);
fprintf(" The model for PCA with best accuracy = %f when m = %d \n" ,acc_pca,i_pca)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%     FS     %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_set_fs = data_set;
[ conf_mats_FS ] = MyFSelection( data_set_fs );

accuracy_FS = zeros(1,57);
precision_FS = zeros(1,57);
recall_FS = zeros(1,57);

for i = 1:57
    conf_mat = conf_mats_FS(:,:,i);
    accuracy_FS(i) = ( conf_mat(1,1) + conf_mat(2,2) ) / sum(sum(conf_mat));
    precision_FS(i) = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
    recall_FS(i) = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
end
m =1:57;
figure
plot(m,accuracy_FS)
title('Accuracy for FS')

figure
plot(m,precision_FS)
title('Precision for FS')

figure
plot(m,recall_FS)
title('Recall for FS')


conf_mat = conf_mats_FS(:,:,2);
accuracy_FS_2 = ( conf_mat(1,1) + conf_mat(2,2) ) / sum(sum(conf_mat));
precision_FS_2 = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1));
recall_FS_2 = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2));
    
l=[accuracy_FS_2 precision_FS_2 recall_FS_2];
figure
stem(l)
title('accuracy precision recall')

[acc_fs,i]=max(accuracy_FS);
fprintf(" The model with best accuracy = %f when m = %d \n" ,acc_fs,i)
    
    