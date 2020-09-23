function example_5()
% -------------------------------------------------------------------------
% EXAMPLE 5 accompanying Ensemble implementation as described in [4].
% -------------------------------------------------------------------------
% This is the last provided example. We will still steganalyze HUGO [1] at
% payload 0.40 bpp, but this time we will use 1234-dimensional CDF features
% [5] (union of CC-PEV features [2] and SPAM features [6]). Image database
% is BOSSBase v0.92 [3]. The point of this example is to show that instead
% of a single feature set, a multiple of feature-set files may stay at the
% input of the ensemble, i.e. there is no need to concatenate CC-PEV and
% SPAM features manually, ensemble will take care of that. This is
% useful when we have many different sub-feature sets and want to try
% different combinations of them for steganalysis. For the purposes of this
% example, as the full dimensionality of the SPAM feature space is d=1234,
% d_sub is fixed to d_sub = 800. Parameter L will be determined
% automatically. Half of the images is used for training and the other half
% for testing, only a single TRN/TST split this time.
%
% In this example, we also explain other options and functionalities of our
% ensemble implementation. See the comments throughout the code to learn
% more.
%
% Note: The choice of d_sub = 800 is indeed around the optimal value of
% d_sub, so the obtained testing error (should be around 32%-33%)
% corresponds to the security of HUGO under this experimental setup (CDF
% features, image database, payload, TRN/TST ratio etc.). For more accurate
% statistical data, the experiment should be repeated for many different
% TRN/TST splits and/or different PRNG seeds for creating bootstrap samples
% and random subspaces.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com
% -------------------------------------------------------------------------
% References:
% [1] - T. Pevny, T. Filler, and P. Bas. Using high-dimensional image
%       models to perform highly undetectable steganography. In R. Bohme
%       and R. Safavi-Naini, editors, Information Hiding, 12th
%       International Workshop, volume 6387 of Lecture Notes in Computer
%       Science, pages 161?77, Calgary, Canada, June 28?0, 2010.
%       Springer-Verlag, New York.
% [2] - J. Kodovsky and J. Fridrich. Calibration revisited. In J. Dittmann,
%       S. Craver, and J. Fridrich, editors, Proceedings of the 11th ACM
%       Multimedia & Security Workshop, pages 63?4, Princeton, NJ,
%       September 7?, 2009.
% [3] - Available at http://www.agents.cz/boss/BOSSFinal/ in the Materials
%       section.
% [4] - J. Kodovsky, J. Fridrich, and V. Holub. Ensemble classifiers for
%       steganalysis of digital media. IEEE Transactions on Information
%       Forensics and Security. Currently under review.
% [5] - J. Kodovsky, T. Pevny, and J. Fridrich. Modern steganalysis can
%       detect YASS. In N. D. Memon, E. J. Delp, P. W. Wong, and J.
%       Dittmann, editors, Proceedings SPIE, Electronic Imaging, Security
%       and Forensics of Multimedia XII, volume 7541, pages 02?1?2?1,
%       San Jose, CA, January 17?1, 2010.
% [6] - T. Pevny, P. Bas, and J. Fridrich. Steganalysis by subtractive
%       pixel adjacency matrix. IEEE Transactions on Information Forensics
%       and Security, 5(2):215?24, June 2010.
% -------------------------------------------------------------------------

%%% <ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Location of extracted cover and stego features, in this case CC-PEV
% features and SPAM features. IMPORTANT: the order of the feature-set files
% for cover and stego must be identical!
settings.cover = {'fea/BOSSbase/cover/ccpev.mat','fea/BOSSbase/cover/spam_2nd_T3.mat'};
settings.stego = {'fea/BOSSbase/HUGO_40/ccpev.mat','fea/BOSSbase/HUGO_40/spam_2nd_T3.mat'};
%两种特征合成一个特征
% PRNG seed for TRN/TST split needs to be specified, there is no default value
settings.seed_trntst = 712; % PRNG seed for the TRN/TST split

% PRNG seeds for generating random subspaces and bootstrap samples are not
% required at the input (by default they are determined randomly). However,
% if you want to generate EXACTLY the same random subspaces and EXACTLY the
% same bootstrap samples (reproducibily of the same results), you can
% specify the following two PRNG seeds:
settings.seed_subspaces = 338;  % PRNG seed for random subspaces
settings.seed_bootstrap = 683;  % PRNG seed for bootstrap samples

% By default, the progress of the ensemble classification, as well as its
% results, are stored in the directory './output'. The name of the output
% file is generated automatically. You can change the output logfile by
% setting the parameter settings.output = 'DESIRED_OUTPUT_FILE';

% Additionally, the progress of the ensemble classification is displayed in
% the Matlab's command window. This can be suppressed by uncommenting the
% following line:
% settings.verbose = 0;

% The relative number of training images (TRN/TST ratio) is by default set
% to 0.5 and thus does not have to be specified when half of the images
% are to be used for training and the other half for testing. The following
% line is thus commented.
% settings.ratio = 0.5; % Relative number of training images

settings.d_sub = 800; % Fixed value of d_sub (random subspace dimensionality)

% If you want to avoid the time consuming search, you can fix d_sub to a
% pre-defined value (as we did to d_sub = 800 in this example). However, by
% default both d_sub and L are automatically determined by the techniques
% described in [4].
%
% The default parameters of the search for d_sub are as follows.
%
%   settings.k_step = 200;            % initial step for d_sub when searching from left (stage 1 of Algorithm 2 in [4])
%   settings.Eoob_tolerance = 0.02;   % The relative tolerance for the minimality of OOB within the search, i.e. specifies the stopping criterion for the stage 2 in Algorithm 2
%
% The default parameters of the automatic stopping criterion for L are set
% as follows:
%
%   settings.L_kernel = ones(1,5)/5;  % moving average of OOB estimates taken over 5 values
%   settings.L_min_length = 25;       % at least 25 random subspaces will be generated every time
%   settings.L_memory = 50;           % last 50 OOB estimates need to stay in the epsilon tube
%   settings.L_epsilon = 0.005;       % specification of the epsilon tube
%
% According to our experiments, these values are sufficient for most of the
% steganalysis tasks (different algorithms and features). Nevertheless, any
% of these parameters can be modified before calling the ensemble if
% desired.
%
% See our publication [4] or directly the code of ensemble for more details
% about the actual implementation of the search for d_sub and the stopping
% criterion for L.

% The maximum number of base learners is by default set to 500. Once this
% number is reached, the training will stop even if OOB error hasn't
% converged yet. You can increase (or decrease) this default maximum number
% of base learners by specifying
% settings.max_number_base_learners = DESIRED_VALUE;

%%% </ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ensemble(settings); % launch ENSEMBLE classifier
