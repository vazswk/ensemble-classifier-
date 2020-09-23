function example_4()
% -------------------------------------------------------------------------
% EXAMPLE 4 accompanying Ensemble implementation as described in [4].
% -------------------------------------------------------------------------
% We will perform steganalysis of HUGO [1] at payload 0.40 bpp using
% 548-dimensional CC-PEV features [2] using BOSSBase v0.92 [3] image
% database, as we did in all the previous examples. However, this time both
% d_sub and L will be fixed. The random subspace dimensionality is fixed to
% d_sub = 100, and L is fixed to 200, a relatively large value.
% Bootstrapping of training samples will be turned off as it is no longer 
% needed for the search for d_sub, neither for the automatic stopping
% criterion for L.
%
% Experimental setup: As in the example 3, we will use 80% of the images
% for training and the other 20% for testing, and will repeat the whole
% procedure for 10 different splits into training and testing sets. Average
% testing error will be reported.
%
% Note: The resulting average error over 10 splits should be roughly the
% same as in example 3. Increasing L to 200 doesn't improve the
% performance, the default parameters of the automatic stopping criterion
% for L used in example 3 are sufficient for a good performance.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com
% -------------------------------------------------------------------------
% References:
% [1] - T. Pevny, T. Filler, and P. Bas. Using high-dimensional image
%       models to perform highly undetectable steganography. In R. Bohme
%       and R. Safavi-Naini, editors, Information Hiding, 12th
%       International Workshop, volume 6387 of Lecture Notes in Computer
%       Science, pages 161–177, Calgary, Canada, June 28–30, 2010.
%       Springer-Verlag, New York.
% [2] - J. Kodovsky and J. Fridrich. Calibration revisited. In J. Dittmann,
%       S. Craver, and J. Fridrich, editors, Proceedings of the 11th ACM
%       Multimedia & Security Workshop, pages 63–74, Princeton, NJ,
%       September 7–8, 2009.
% [3] - Available at http://www.agents.cz/boss/BOSSFinal/ in the Materials
%       section.
% [4] - J. Kodovsky, J. Fridrich, and V. Holub. Ensemble classifiers for
%       steganalysis of digital media. IEEE Transactions on Information
%       Forensics and Security. Currently under review.
% -------------------------------------------------------------------------

%%% <ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

settings.cover = 'fea/BOSSbase/cover/ccpev.mat'; % extracted cover CC-PEV features
settings.stego = 'fea/BOSSbase/HUGO_40/ccpev.mat'; % extracted stego CC-PEV features

settings.d_sub = 100;         % Fixed value of d_sub (random subspace dimensionality)
settings.L = 200;             % Fixed value of L (number of random subspaces / base learners)
settings.ratio = 0.8;         % Relative number of training images

% Since both d_sub and L are fixed, we can turn off forming of bootstrap
% samples. In that case, each base learner is trained on the whole training
% part of the image database. If you still want to perform bootstrapping,
% comment the following line
settings.bootstrap = 0; % Turn off bootstrapping of training samples

number_of_splits = 10; % number of TRN/TST splits

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
