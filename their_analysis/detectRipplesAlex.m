function [rippleIdx, rippleSnips] = detectRipplesAlex(freqFilter,lfpSignal,runSignal,fs)

%detectRipples is used to automatically detect peaks in the ripple
%frequency and extract indices of these peaks so that snippets of the LFP
%signals containing putative ripples may be plotted for visual verification.

%%INPUTS: 
%freqFilter: variable containing the upper and lower frequency limits of
%the relevant ripple range in Hz, e.g. [150 250]
%
%lfpSignal: 1 x N vector containing the LFP signal, with N = number of
%samples
%
%runSignal: running speed, measured from wheel signal
%
%fs: sampling frequency in Hz, e.g. 2500

%%OUTPUTS:
%rippleLocs: contains the indices (sample number from LFP signal) of
%putative ripple peaks

%rippleSnips: struct containing snippets of the LFP signal centered on
%putative ripples

rawLFP = lfpSignal;

nSnips = floor(length(rawLFP)/(fs)) - 1;
time = linspace(0,length(rawLFP),length(rawLFP))/(fs);
timeRound = round(time,3);
rippleLocs = [];


LFP = rawLFP;




% Filter LFP between 150-250 hz for sharp wave ripple detection
freqL = freqFilter(1);
freqU = freqFilter(2);
nyquistFs = fs/2;
%min_ripple_width = 0.015; % minimum width of envelop at upper threshold for ripple detection in ms
    
% Thresholds for ripple detection 
U_threshold = 3;  % in standard deviations
% L_threshold = 1; % in standard deviations


% Create filter and apply to LFP data
filter_kernel = fir1(600,[freqL freqU]./nyquistFs); % Different filters can also be tested her e.g. butter and firls

filtered_lfp = filtfilt(filter_kernel,1,LFP); % Filter LFP using the above created filter kernel