function  [testing_errors, testing_error, std_testing_error]= my_tutorial_for_test(cover_path_train,stego_path_train,cover_path_test,stego_path_test)%,name_ensemble) 
%%%%%%%%%%%  train   %%%%%%%%%%
cover_train = load(cover_path_train);
stego_train = load(stego_path_train);
%% 为了避免载体和隐写体数目不同，求其交集
names_train = intersect(cover_train.names,stego_train.names); 
names_train = sort(names_train);   %为了避免载体和隐写体顺序不同，对其进行重新排序
%% 分别在cover和stego中选择共有的样本,按照名称进行排序，从而将stego与cover的特征一一对应上
cover_train_names = cover_train.names(ismember(cover_train.names,names_train));
[cover_train_names,ix] = sort(cover_train_names); %对特征样本进行排序，得到排序后的名称和序号
C_train = cover_train.F(ismember(cover_train.names,names_train),:);  %ismember返回的是坐标
C_train = C_train(ix,:);  %使得载体特征样本的名称names和数值F一一对应
% Prepare stego features S
stego_train_names = stego_train.names(ismember(stego_train.names,names_train));
[stego_train_names ,ix] = sort(stego_train_names);
S_train = stego_train.F(ismember(stego_train.names,names_train),:);
S_train = S_train(ix,:); %使得隐写体特征样本的名称names和数值F一一对应


%%%%%%%%%%%  test   %%%%%%%%%%
cover_test = load(cover_path_test);
stego_test = load(stego_path_test);
%% 为了避免载体和隐写体数目不同，求其交集
names_test = intersect(cover_test.names,stego_test.names); 
names_test = sort(names_test);   %为了避免载体和隐写体顺序不同，对其进行重新排序
%% 分别在cover和stego中选择共有的样本,按照名称进行排序，从而将stego与cover的特征一一对应上
cover_test_names = cover_test.names(ismember(cover_test.names,names_test));
[cover_test_names,ix] = sort(cover_test_names); %对特征样本进行排序，得到排序后的名称和序号
C_test = cover_test.F(ismember(cover_test.names,names_test),:);  %ismember返回的是坐标
C_test = C_test(ix,:);  %使得载体特征样本的名称names和数值F一一对应
% Prepare stego features S
stego_test_names = stego_test.names(ismember(stego_test.names,names_test));
[stego_test_names ,ix] = sort(stego_test_names);
S_test = stego_test.F(ismember(stego_test.names,names_test),:);
S_test = S_test(ix,:); %使得隐写体特征样本的名称names和数值F一一对应

for seed = 1:10
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed)); %创建伪随机序列
    training_set = randperm(size(C_train,1));
    % Prepare training features
    TRN_cover = C_train(training_set,:);
    TRN_stego = S_train(training_set,:);
    
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed)); %创建伪随机序列
    testing_set = randperm(size(C_test,1));  
    % Prepare testing features
    TST_cover = C_test(testing_set,:);
    TST_stego = S_test(testing_set,:);
    
    [trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego);
    test_results_cover = ensemble_testing(TST_cover,trained_ensemble);     
    test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
    false_alarms = sum(test_results_cover.predictions~=-1);
    missed_detections = sum(test_results_stego.predictions~=+1);   
    num_testing_samples = size(TST_cover,1)+size(TST_stego,1);
    testing_errors(seed) = (false_alarms + missed_detections)/num_testing_samples;
    fprintf('Testing error %i: %.4f\n',seed,testing_errors(seed));
    %% 将分类错误的样本名进行存储
    TST_cover_names = cover_test_names(testing_set);
    TST_stego_names = stego_test_names(testing_set);
    cover_false_alarms = TST_cover_names((test_results_cover.predictions~=-1));  %载体被错误警报的样本
    stego_missed_detections = TST_stego_names((test_results_stego.predictions~=+1));  %隐写体没有被警报的样本
end
fprintf('---\nAverage testing error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors),std(testing_errors));
testing_error = mean(testing_errors);
std_testing_error = std(testing_errors);

% xlswrite([name_ensemble '_cover_false_alarms.xls'],cover_false_alarms);
% xlswrite([name_ensemble '_stego_missed_detections.xls'],stego_missed_detections);
%save('C:\Users\VAZ\Desktop\jpeg ca\test_new_results\test_new_results\comp_200-num-1500\pev\pev_H\1.txt','testing_error')


