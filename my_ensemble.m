function test_error = my_ensemble(cover,stego)
%cover = 'C:\Users\VAZ\Desktop\jpeg ca\test_results_STC\feature extraction\ccJRM\cover';
%stego = 'C:\Users\VAZ\Desktop\jpeg ca\test_results_STC\feature extraction\ccJRM\AVG_Distribution_stego_rayligh_10';
%%% <ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num=10；
settings.cover = cover; % extracted cover CC-PEV features
settings.stego = stego; % extracted stego CC-PEV features
test_errors = zeros(1,num);
for i = 1:num
    i
     settings.seed_trntst = rand(1)*10000; % PRNG seed for the TRN/TST split (no default value, needs to be specified)

%      settings.d_sub = 100;         % Fixed value of d_sub (random subspace dimensionality)
%      settings.L = 200;             % Fixed value of L (number of random subspaces / base learners)
%      settings.ratio = 0.8;
%      settings.verbose = 0;
%      settings.bootstrap = 0; % Turn off bootstrapping of training samples
    %%% </ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    result = ensemble(settings); % launch ENSEMBLE classifier
    test_errors(i) = result.testing_error;
end
test_error = mean(test_errors);
