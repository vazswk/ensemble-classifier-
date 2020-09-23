function example_3()
% -------------------------------------------------------------------------
% EXAMPLE 3 accompanying Ensemble implementation as described in [4].
% -------------------------------------------------------------------------
% In this example, the search for d_sub will be turned off and fixed to a
% pre-defined value. This speeds up the feature development cycle once the
% steganalyst gets a feeling about the optimal value of d_sub. We will
% stick to the steganalysis of HUGO [1] at payload 0.40 bpp and to the
% 548-dimensional CC-PEV feature space [2] as we did in examples 1 and 2.
% The random subspace dimensionality is fixed to d_sub = 100. We will still
% use the automatic stopping criterion for L. Therefore, out-of-bag (OOB)
% error estimates (and thus bootstrapping of training samples) are needed,
% see [4] for more details.
% 
% Experimental setup: This time, we will use 80% of the images for training
% and the other 20% for testing, randomly divided. We want to repeat this
% procedure for 10 different splits into training and testing sets and
% report the average error.
% 
% Note: The resulting testing error is higher than in examples 1 and 2
% (should be around 42%) as the random subspace dimensionality d_sub = 100
% is sub-optimal (compare to the found optimal values of d_sub from example
% 1 or 2). Even the increased number of training samples (80% vs. 50%)
% did not help to overcome this sub-optimality of the choice of d_sub.
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
settings.d_sub = 100; % Fixed value of d_sub (random subspace dimensionality)
settings.ratio = 0.8; % Relative number of training images (default = 0.5)

% Covariance memory caching is turned off, it does not make sense as no
% search for d_sub is performed. Thus the following line is commented:
% settings.keep_cov = 1;

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
