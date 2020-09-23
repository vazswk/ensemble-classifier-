clc
clear

cover_path = 'e:\pf_testT\cover_T10.dat';
stego_path = 'e:\pf_testT\stego_10_T10.dat';
vecnum = 2646;
image_num = 1000;

names = cell(image_num,1);
for i=1:image_num
    tmp = [num2str(i) '.jpg'];
    names{i,1} = tmp;
end;
fp_feather = fopen(cover_path,'r');
[F,count] = fread(fp_feather,[vecnum,inf],'double');
F = single(F');
if exist('cover.mat','dir');delete cover.mat;end
save cover F names;
fp_feather = fopen(stego_path,'r');
[F,count] = fread(fp_feather,[vecnum,inf],'double');
F = single(F');
if exist('stego.mat','dir');delete stego.mat;end
save stego F names;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
settings.cover = 'cover.mat'; % extracted cover CC-PEV features
settings.stego = 'stego.mat'; % extracted stego CC-PEV features
%settings.seed_trntst = 1; % PRNG seed for the TRN/TST split (no default value, needs to be specified)
%settings.d_sub = 10;
%settings.L = 1;

% PRNG seeds for generating random subspaces and bootstrap samples are not
% required at the input (by default they are determined randomly). However,
% if you want to generate EXACTLY the same random subspaces and EXACTLY the
% same bootstrap samples (reproducibily of the same results), you can
% specify the following two PRNG seeds:
settings.seed_subspaces = 338;  % PRNG seed for random subspaces
settings.seed_bootstrap = 683;  % PRNG seed for bootstrap samples

number_of_splits = 3; % number of TRN/TST splits

%settings.ratio = 0.8;%两种特征合成一个特征
%settings.cover = {'fea/BOSSbase/cover/ccpev.mat','fea/BOSSbase/cover/spam_2nd_T3.mat'};
%settings.stego = {'fea/BOSSbase/HUGO_40/ccpev.mat','fea/BOSSbase/HUGO_40/spam_2nd_T3.mat'};
%%% </ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize final results structure
results = cell(number_of_splits,1);

% loop over all TRN/TST splits
for i=1:number_of_splits
    settings.seed_trntst = i; % PRNG seed for i-th TRN/TST split
    results{i} = ensemble(settings); % launch ENSEMBLE classifier
end

% calculate and report mean error over all TRN/TST splits
results_all = zeros(number_of_splits,1); for i=1:number_of_splits, results_all(i) = results{i}.testing_error; end
fprintf('# -------------------------\nAVERAGE TESTING ERROR OVER %i splits: %.4f\n',number_of_splits,mean(results_all));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delete cover.mat
delete stego.mat