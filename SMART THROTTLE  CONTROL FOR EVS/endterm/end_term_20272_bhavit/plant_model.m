%% plant_model.m — Open-Loop Motor Transfer Function
% EV Throttle Control | End-Term Project | Bhavit Meena (240272)
% 
% Purpose:
%   Models the throttle-to-motor-speed plant as a first-order transfer
%   function and plots the open-loop step response (no controller attached).
%
% Plant Model:
%          K
%  G(s) = -----     K = steady-state gain, tau = time constant
%         τs + 1

clear; clc; close all;

%% --- Plant Parameters ---
K   = 1;     % Steady-state gain  [rpm / (throttle unit)]
tau = 0.5;   % Time constant      [seconds]

%% --- Define Transfer Function ---
G = tf(K, [tau 1]);
fprintf('Plant Transfer Function:\n');
disp(G);

%% --- Open-Loop Step Response ---
t = 0:0.01:5;   % Time vector: 0 to 5 seconds

figure('Name','Open-Loop Motor Step Response','NumberTitle','off');
[y, t_out] = step(G, t);
plot(t_out, y, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Motor Speed (normalised)');
title('Open-Loop Motor Step Response');
legend('Plant G(s): K=1, \tau=0.5 s', 'Location', 'southeast');

% Annotate settling time
info = stepinfo(G);
fprintf('\n--- Open-Loop Step Response Info ---\n');
fprintf('  Rise Time      : %.4f s\n', info.RiseTime);
fprintf('  Settling Time  : %.4f s\n', info.SettlingTime);
fprintf('  Overshoot      : %.2f %%\n', info.Overshoot);
fprintf('  Steady-State   : %.4f\n', dcgain(G));

%% --- Save Plot ---
saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'results', 'open_loop_response.png'));
fprintf('\nSaved: results/open_loop_response.png\n');
