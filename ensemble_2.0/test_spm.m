clc;clear;
load('Secrue_samples_sweep_20.mat')
N = numel(name)
load('sort_name')

for i = 1:N
    name1(i) = str2num(name{i});
    name2(i) = str2num(file_name_sort{i});
end

Spearman_corr = corr(name1', name2', 'type' , 'Spearman')