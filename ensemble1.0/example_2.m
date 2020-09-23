function example_2()
% -------------------------------------------------------------------------
% EXAMPLE 2 accompanying Ensemble implementation as described in [4].
% -------------------------------------------------------------------------
% Typically, steganalyst wants to repeat the experiment several times for
% different splits of the original database into training and testing
% parts. In this example, we will perform the very same task as in example
% number 1, i.e. we will steganalyze HUGO [1] at payload 0.40 bpp using
% 548-dimensional CC-PEV features [2] using BOSSBase v0.92 [3] image
% database. However, we will repeat the experiment for three different
% TRN/TST splits (half/half) and report the average error achieved.
%
% Ensemble setup: Fully automatized ensemble as described in [4]. In this
% example, we turn on the caching of covariance matrices during the search
% for d_sub (keeps the previously calculated cov. matrices in memory and
% uses them for the next value of d_sub). This results in a speed-up of the
% search, especially for large-dimensional features, but may require a lot
% of computer memory. Therefore, in case of memory issues, disable this
% option by commenting an appropriate line in the code below.
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

% covariance memory caching turned on => speed-up (may require a lot of
% memory; in case of MEMORY issues, disable this by commenting the
% following line)
settings.keep_cov = 1;

number_of_splits = 3; % number of TRN/TST splits

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
