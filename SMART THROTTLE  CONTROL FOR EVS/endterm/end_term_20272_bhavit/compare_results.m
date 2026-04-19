%% compare_results.m — Side-by-Side Comparison of All Controllers
% EV Throttle Control | End-Term Project | Bhavit Meena (240272)
%
% Purpose:
%   Overlays the open-loop plant response, PID closed-loop response, and
%   gain-scheduled response on a single figure for comparison.

clear;
clc;
close all;

% % -- -Plant-- - K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

% % -- -PID(Zone 2 / mid gains = reference single - PID)-- - Kp = 2.5;
Ki = 1.0;
Kd = 0.1;
C_pid = pid(Kp, Ki, Kd);
T_pid = feedback(C_pid * G, 1);

%% --- Gain-Scheduled: mid-zone block (Zone 2) for fair comparison ---
Kp_gs = 2.5;
Ki_gs = 1.0;
Kd_gs = 0.10;
C_gs = pid(Kp_gs, Ki_gs, Kd_gs);
T_gs = feedback(C_gs * G, 1);

% % -- -Time Vector-- - t = 0 : 0.01 : 5;

% % -- -Step Responses-- - [ y_ol, t_ol ] = step(G, t);
[ y_pid, t_pid ] = step(T_pid, t);
[ y_gs, t_gs ] = step(T_gs, t);

% % -- -Comparison Plot-- -
    figure('Name', 'Throttle–Motor Response Comparison', 'NumberTitle', 'off');
hold on;
plot(t_ol, y_ol, 'r--', 'LineWidth', 2, 'DisplayName',
     'Open-Loop (no controller)');
plot(t_pid, y_pid, 'b-', 'LineWidth', 2, 'DisplayName',
     sprintf('PID  (Kp=%.1f, Ki=%.1f, Kd=%.1f)', Kp, Ki, Kd));
plot(t_gs, y_gs, 'g-', 'LineWidth', 2, 'DisplayName',
     'Gain-Scheduled (Zone 2 gains)');
yline(1.0, 'k:', 'Setpoint', 'LineWidth', 1);
hold off;
grid on;
xlabel('Time (s)');
ylabel('Motor Speed (normalised)');
title('Throttle–Motor Response Comparison');
legend('Location', 'southeast');

% % -- -Print Summary Table-- - info_ol = stepinfo(G);
info_pid = stepinfo(T_pid);
info_gs = stepinfo(T_gs);

fprintf('\n--- Performance Comparison Table ---\n');
fprintf('%-25s %12s %12s %12s\n', 'Metric', 'Open-Loop', 'PID', 'Gain-Sched');
fprintf('%-25s %12.4f %12.4f %12.4f\n', 'Rise Time (s)', info_ol.RiseTime,
        info_pid.RiseTime, info_gs.RiseTime);
fprintf('%-25s %12.4f %12.4f %12.4f\n', 'Settling Time (s)',
        info_ol.SettlingTime, info_pid.SettlingTime, info_gs.SettlingTime);
fprintf('%-25s %12.2f %12.2f %12.2f\n', 'Overshoot (%%)', info_ol.Overshoot,
        info_pid.Overshoot, info_gs.Overshoot);
fprintf('%-25s %12.4f %12.4f %12.4f\n', 'DC Gain', dcgain(G), dcgain(T_pid),
        dcgain(T_gs));

% % -- -Save-- - saveas(gcf, fullfile(fileparts(mfilename('fullpath')),
                                      'results', 'comparison_plot.png'));
fprintf('\nSaved: results/comparison_plot.png\n');
