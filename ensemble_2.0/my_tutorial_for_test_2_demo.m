clc;clear;
cover = 'CMU_1s_2000_MFCC_cover.mat';
stego = 'CMU_1s_2000_MFCC_stego_0.3.mat';
times = 20;
for sweeps = 20
tic
test_num = times*sweeps/2;
% sweeps = 50;  % 每张图参与测试的次数为：times*sweeps/2
[testing_errors, testing_error, std_testing_error,test_wrong_names]= my_tutorial_for_test_2(cover,stego,sweeps,times);
save('wrong_name', 'test_wrong_names')
for split = 1:sweeps
    for i = 1:times
        A = test_wrong_names{split,i,1};
        B = test_wrong_names{split,i,2};
        if (split*i ==1)
           % cover_false_alarms
           T = A;
           % stego_missed_detections
           L = numel(T);
           for num = 1:numel(B)
               T{L+num} = B{num};
           end
        else
           % cover_false_alarms
           L = numel(T);
           for num = 1:numel(A)
               T{L+num} = A{num};
           end
           % stego_missed_detections
           L = numel(T);
           for num = 1:numel(B)
               T{L+num} = B{num};
           end
        end
    end
end

Freq = tabulate(T);

[H,W] = size(Freq);
name = cell(H,1);
count = zeros(H,1);
for i = 1:H
    name{i}= Freq{i};
    count(i) = Freq{H+i};
end
[count, index] = sort(count,'descend');
name = name(index);
toc
path = ['Secrue_samples_sweep_' num2str(sweeps)];
save(path,'count','name','-v7.3');
end

