function example_1()
% -------------------------------------------------------------------------
% EXAMPLE 1 accompanying Ensemble implementation as described in [4].
% -------------------------------------------------------------------------
% In this example, we will steganalyze HUGO [1] at payload 0.40 bpp using
% 548-dimensional CC-PEV features [2]. Image database: BOSSBase v0.92 [3].
% Half of the images are used for training, and the other half for testing,
% randomly divided. Experiment will be performed only once, i.e. for a
% single TRN/TST split.
%
% This example shows the most simple (and the most common) way of using the
% ensemble - fully automatized search for d_sub and the automatic stopping
% criterion for L, the number of random subspaces, both based on the
% technique of out-of-bag (OOB) error estimates described in [4]. All the
% parameters are default. As can be seen from the code below, the only
% needed input are cover/stego feature files and the TRN/TST splitting PRNG
% seed. The progress of the training will be printed into Matlab's command
% window. 
%
% Note: The obtained testing error should be around 39% which is the
% performance of CC-PEV features when attacking HUGO under this
% experimental setup (image database, payload, TRN/TST ratio etc.).
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
% -------------------------------------------------------------------------

%%% <ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

settings.cover = 'fea/BOSSbase/cover/ccpev.mat'; % extracted cover CC-PEV features
settings.stego = 'fea/BOSSbase/HUGO_40/ccpev.mat'; % extracted stego CC-PEV features
settings.cover = 'cover.mat'; % extracted cover CC-PEV features
settings.stego = 'stego.mat'; % extracted stego CC-PEV features
settings.seed_trntst = 1; % PRNG seed for the TRN/TST split (no default value, needs to be specified)
settings.d_sub = 10;
settings.L = 1;
%%% </ENSEMBLE SETUP> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ensemble(settings); % launch ENSEMBLE classifier
