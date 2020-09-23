function  [testing_errors, testing_error, std_testing_error]= my_tutorial_for_test(cover_path_train,stego_path_train,cover_path_test,stego_path_test)%,name_ensemble) 
%%%%%%%%%%%  train   %%%%%%%%%%
cover_train = load(cover_path_train);
stego_train = load(stego_path_train);
%% Ϊ�˱����������д����Ŀ��ͬ�����佻��
names_train = intersect(cover_train.names,stego_train.names); 
names_train = sort(names_train);   %Ϊ�˱����������д��˳��ͬ�����������������
%% �ֱ���cover��stego��ѡ���е�����,�������ƽ������򣬴Ӷ���stego��cover������һһ��Ӧ��
cover_train_names = cover_train.names(ismember(cover_train.names,names_train));
[cover_train_names,ix] = sort(cover_train_names); %�����������������򣬵õ����������ƺ����
C_train = cover_train.F(ismember(cover_train.names,names_train),:);  %ismember���ص�������
C_train = C_train(ix,:);  %ʹ��������������������names����ֵFһһ��Ӧ
% Prepare stego features S
stego_train_names = stego_train.names(ismember(stego_train.names,names_train));
[stego_train_names ,ix] = sort(stego_train_names);
S_train = stego_train.F(ismember(stego_train.names,names_train),:);
S_train = S_train(ix,:); %ʹ����д����������������names����ֵFһһ��Ӧ


%%%%%%%%%%%  test   %%%%%%%%%%
cover_test = load(cover_path_test);
stego_test = load(stego_path_test);
%% Ϊ�˱����������д����Ŀ��ͬ�����佻��
names_test = intersect(cover_test.names,stego_test.names); 
names_test = sort(names_test);   %Ϊ�˱����������д��˳��ͬ�����������������
%% �ֱ���cover��stego��ѡ���е�����,�������ƽ������򣬴Ӷ���stego��cover������һһ��Ӧ��
cover_test_names = cover_test.names(ismember(cover_test.names,names_test));
[cover_test_names,ix] = sort(cover_test_names); %�����������������򣬵õ����������ƺ����
C_test = cover_test.F(ismember(cover_test.names,names_test),:);  %ismember���ص�������
C_test = C_test(ix,:);  %ʹ��������������������names����ֵFһһ��Ӧ
% Prepare stego features S
stego_test_names = stego_test.names(ismember(stego_test.names,names_test));
[stego_test_names ,ix] = sort(stego_test_names);
S_test = stego_test.F(ismember(stego_test.names,names_test),:);
S_test = S_test(ix,:); %ʹ����д����������������names����ֵFһһ��Ӧ

for seed = 1:10
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed)); %����α�������
    training_set = randperm(size(C_train,1));
    % Prepare training features
    TRN_cover = C_train(training_set,:);
    TRN_stego = S_train(training_set,:);
    
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed)); %����α�������
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
    %% �������������������д洢
    TST_cover_names = cover_test_names(testing_set);
    TST_stego_names = stego_test_names(testing_set);
    cover_false_alarms = TST_cover_names((test_results_cover.predictions~=-1));  %���屻���󾯱�������
    stego_missed_detections = TST_stego_names((test_results_stego.predictions~=+1));  %��д��û�б�����������
end
fprintf('---\nAverage testing error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors),std(testing_errors));
testing_error = mean(testing_errors);
std_testing_error = std(testing_errors);

% xlswrite([name_ensemble '_cover_false_alarms.xls'],cover_false_alarms);
% xlswrite([name_ensemble '_stego_missed_detections.xls'],stego_missed_detections);
%save('C:\Users\VAZ\Desktop\jpeg ca\test_new_results\test_new_results\comp_200-num-1500\pev\pev_H\1.txt','testing_error')


