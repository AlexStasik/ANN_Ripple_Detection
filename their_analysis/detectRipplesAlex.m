function [rippleIdx, rippleSnips, rippleLocs, w] = detectRipplesAlex(freqFilter, lfpSignal, runSignal, fs)

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

lfp = lfpSignal;
rawLFP = lfpSignal;

nSnips = floor(length(rawLFP)/(fs)) - 1;
time = linspace(0,length(lfp),length(lfp))/(fs);
timeRound = round(time,3);
rippleLocs = [];

window_size = 1;
window_size_index = window_size * fs;


    
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

filtered_lfp = filtfilt(filter_kernel,1,lfp); % Filter LFP using the above created filter kernel

% Hilbert transform LFP to calculate envelope
lfp_hil_tf = hilbert(filtered_lfp);
lfp_envelop = abs(lfp_hil_tf);

% Smooth envelop using code from 
% https://se.mathworks.com/matlabcentral/fileexchange/43182-gaussian-smoothing-filter?focused=3839183&tab=function 
smoothed_envelop = gaussfilt_2017(time,lfp_envelop,.004);
moving_mean = movmean(smoothed_envelop, window_size_index);
moving_std = movstd(smoothed_envelop, window_size_index);
moving_mean_move = movmean(runSignal, window_size_index);


% Find upper/lower threshold values of the LFP
upper_thresh = moving_mean + U_threshold*moving_std;
    

% Find peaks of envelop. NB: The parameters of this function have to be properly
% chosen for best result.
[~,locs,w,~] = findpeaks(smoothed_envelop-upper_thresh,fs,'MinPeakHeight',0,'MinPeakDistance',0.025,'MinPeakWidth',0.015,'WidthReference','halfheight','Annotate','extents','WidthReference','halfprom');
rippleLocs = round(locs,3);    
 

    
% end

rippleSnips = struct();
rippleIdx = zeros(1,length(rippleLocs));
rippleLocs = round(rippleLocs,3);
%convert the ripple locations from time to sample
for i = 1:length(rippleLocs)
        lfpPeakIdx = find(timeRound == rippleLocs(i));
        lfpStartIdx = lfpPeakIdx(1) - (0.5*fs);
        %if the ripple timepoint is near the beginning of the trace
        if lfpStartIdx < 0; lfpStartIdx = 1; end
        lfpEndIdx = lfpPeakIdx(1) + (0.5*fs);
        %if the ripple timepoint is near the end of the trace
        if lfpEndIdx > length(lfpSignal); lfpEndIdx = length(lfpSignal); end
        rippleSnips(i).lfp = rawLFP(lfpStartIdx:lfpEndIdx);
        rippleIdx(i) = lfpPeakIdx(1);
    if runSignal(rippleIdx(i)) ~= 0; rippleIdx(i) = NaN; end %take out timepoints when animal is walking
%     if runSignal(moving_mean_move(i)) ~= 0; rippleIdx(i) = NaN; end %take out timepoints when animal is walking
    
end

%remove NaNs and timepoints too close together (likely identifying the same ripple
%waveform)
rippleSnips(isnan(rippleIdx)) = [];
rippleIdx(isnan(rippleIdx)) = [];
% timeBetweenRipples = diff(rippleIdx);
% %must have at least 100 ms between ripples, if there are 2 close together,
% %keep the later one
% extraRipples = find(timeBetweenRipples < 250);
% rippleIdx(extraRipples + 1) = [];
% rippleSnips(extraRipples + 1) = [];





