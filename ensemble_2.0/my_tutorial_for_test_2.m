function  [testing_errors, testing_error, std_testing_error,test_wrong_names]= my_tutorial_for_test_2(cover_path,stego_path,sweeps,times)%,name_ensemble) 
%% �����޸�ʵ�֣� PE = ��P_FA + P_MD��/2
%%%%%%%%%%%  train   %%%%%%%%%%
cover = load(cover_path);
stego = load(stego_path);
testing_errors = zeros(sweeps,times);
test_wrong_names = cell(sweeps,times,2);

for split = 1:sweeps
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',floor((split-1)/2))); % 0 0 1 1 2 2 3 3 
    random_permutation = randperm(numel(cover.names));
    %% ��������Ŀ���Ǳ�֤ÿ�β����У�cover��stego������ѵ���Ͳ��Ե��������ظ�ʹ�ã�����ȫ����
    if mod(split,2)==1 
       train_index = random_permutation(1:round(length(random_permutation)/2));
       test_index =  random_permutation(round(length(random_permutation)/2)+1:end);
    else
       test_index = random_permutation(1:round(length(random_permutation)/2));
       train_index =  random_permutation(round(length(random_permutation)/2)+1:end);
    end
    %%%%%%%%%%%  train   %%%%%%%%%%
    cover_train_names = cover.names(train_index);
    stego_train_names = stego.names(train_index);
    cover_train_F = cover.F(train_index,:);
    stego_train_F = stego.F(train_index,:);
    %% Ϊ�˱����������д����Ŀ��ͬ�����佻��
    names_train = intersect(cover_train_names,stego_train_names); 
    names_train = sort(names_train);   %Ϊ�˱����������д��˳��ͬ�����������������
    %% �ֱ���cover��stego��ѡ���е�����,�������ƽ������򣬴Ӷ���stego��cover������һһ��Ӧ��
    cover_train_names_N = cover_train_names(ismember(cover_train_names,names_train));
    C_train = cover_train_F(ismember(cover_train_names,names_train),:);  %ismember���ص�������
    [cover_train_names,ix] = sort(cover_train_names_N); %�����������������򣬵õ����������ƺ����
    C_train = C_train(ix,:);  %ʹ��������������������names����ֵFһһ��Ӧ
    % Prepare stego features S
    stego_train_names_N = stego_train_names(ismember(stego_train_names,names_train));
    S_train = stego_train_F(ismember(stego_train_names,names_train),:);
    [stego_train_names ,ix] = sort(stego_train_names_N);   
    S_train = S_train(ix,:); %ʹ����д����������������names����ֵFһһ��Ӧ

    %%%%%%%%%%%  test   %%%%%%%%%%
    cover_test_names = cover.names(test_index);
    stego_test_names = stego.names(test_index);
    cover_test_F = cover.F(test_index,:);
    stego_test_F = stego.F(test_index,:);
    %% Ϊ�˱����������д����Ŀ��ͬ�����佻��
    names_test = intersect(cover_test_names,stego_test_names); 
    names_test = sort(names_test);   %Ϊ�˱����������д��˳��ͬ�����������������
    %% �ֱ���cover��stego��ѡ���е�����,�������ƽ������򣬴Ӷ���stego��cover������һһ��Ӧ��
    cover_test_names_N = cover_test_names(ismember(cover_test_names,names_test));
    C_test = cover_test_F(ismember(cover_test_names,names_test),:);  %ismember���ص�������
    [cover_test_names,ix] = sort(cover_test_names_N); %�����������������򣬵õ����������ƺ����    
    C_test = C_test(ix,:);  %ʹ��������������������names����ֵFһһ��Ӧ
    % Prepare stego features S
    stego_test_names_N = stego_test_names(ismember(stego_test_names,names_test));
    S_test = stego_test_F(ismember(stego_test_names,names_test),:);
    [stego_test_names ,ix] = sort(stego_test_names_N);
    S_test = S_test(ix,:); %ʹ����д����������������names����ֵFһһ��Ӧ
    
    for seed = 1:times
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
        if seed==1
           [trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego);
        else
           % First, we can fix the random subspace dimensionality (d_sub) in order to
           % avoid the expensive search. This can be useful for a fast researchfeedback.
           settings = struct('d_sub',results.optimal_d_sub,'verbose',2);
           [trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego,settings);
        end
        test_results_cover = ensemble_testing(TST_cover,trained_ensemble);     
        test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
        % cover: ������ 0   % stego ������ 1
        FP = sum(test_results_cover.predictions~=-1);  % ��cover����stego 
        FN = sum(test_results_stego.predictions~=+1); % ��stego����cover 
        TP = sum(test_results_stego.predictions==+1); % ��stego����stego
        % ©������
        MA = FN/(TP + FN);
        %�龯����
        FA = FP/(TP + FP); 
        testing_errors(split,seed) =  (MA + FA)/2;              
        fprintf('Testing error %i: %.4f\n',seed,testing_errors(split,seed));
        %% �������������������д洢
        TST_cover_names = cover_test_names(testing_set);
        TST_stego_names = stego_test_names(testing_set);
        cover_false_alarms = TST_cover_names((test_results_cover.predictions~=-1));  %���屻���󾯱�������
        stego_missed_detections = TST_stego_names((test_results_stego.predictions~=+1));  %��д��û�б�����������
        test_wrong_names{split,seed,1} = cover_false_alarms;
        test_wrong_names{split,seed,2} = stego_missed_detections;

    end
end
fprintf('---\nAverage testing error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors(:)),std(testing_errors(:)));
testing_error = mean(testing_errors(:));
std_testing_error = std(testing_errors(:));

% xlswrite([name_ensemble '_cover_false_alarms.xls'],cover_false_alarms);
% xlswrite([name_ensemble '_stego_missed_detections.xls'],stego_missed_detections);
%save('C:\Users\VAZ\Desktop\jpeg ca\test_new_results\test_new_results\comp_200-num-1500\pev\pev_H\1.txt','testing_error')


