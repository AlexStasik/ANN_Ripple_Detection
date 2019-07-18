clear all
close all
clc

%%
addpath(genpath('/Users/astasik/Desktop/current_projects/ANN_Ripple_Detection/their_analysis/gaussfilt'));
data = open('../data/m4000series_LFP_ripple.mat');
animals = fieldnames(data);

%%
freqFilter = [150 250];
fs = 2500;


%%
for i=1:length(animals)
    v = getfield(data, animals{i});
    lfp = v.lfp;
    speed = v.run_speed;
    true_loc = v.rippleLocs;
    
    time = linspace(0,length(lfp),length(lfp))/(fs);
    [rippleIdx, rippleSnips] = detectRipples(freqFilter,lfp,speed,fs);
       
    
    animals{i}
    a = ismember(true_loc, rippleIdx);
    sum(a) / length(true_loc) * 100
    
    save(animals{i}, 'rippleIdx')
end


%%

rawLFP = lfp;
lfp = rawLFP;


% Filter LFP between 150-250 hz for sharp wave ripple detection
    freqL = freqFilter(1);
    freqU = freqFilter(2);
    nyquistFs = fs/2;
    %min_ripple_width = 0.015; % minimum width of envelop at upper threshold for ripple detection in ms
    
    % Thresholds for ripple detection 
    U_threshold = 3;  % in standard deviations
%     L_threshold = 1; % in standard deviations
    
    % Create filter and apply to LFP data
    filter_kernel = fir1(600,[freqL freqU]./nyquistFs); % Different filters can also be tested her e.g. butter and firls

    filtered_lfp = filtfilt(filter_kernel,1,lfp); % Filter LFP using the above created filter kernel
    
    % Hilbert transform LFP to calculate envelope
    lfp_hil_tf = hilbert(filtered_lfp);
    lfp_envelop = abs(lfp_hil_tf);

    % Smooth envelop using code from 
    % https://se.mathworks.com/matlabcentral/fileexchange/43182-gaussian-smoothing-filter?focused=3839183&tab=function 
    smoothed_envelop = gaussfilt_2017(time,lfp_envelop,.004);
    moving_mean = movmean(smoothed_envelop, 2500);
    moving_std = movstd(smoothed_envelop, 2500);
    upper_thresh = moving_mean + U_threshold*moving_std;
    
    [~,locs,~,~] = findpeaks(smoothed_envelop-upper_thresh,fs,'MinPeakHeight',0,'MinPeakDistance',0.025,'MinPeakWidth',0.015,'WidthReference','halfheight');
    locs = round(locs,3);
    
    
    %%
    figure
    plot(time, smoothed_envelop-upper_thresh)
    
    
    
    %%
    
figure
hold on
plot(time, smoothed_envelop)
plot(time, lfp_envelop)
plot(time, filtered_lfp)
plot(time, moving_mean + U_threshold*moving_std)