% compare_results.m
% Compare open-loop, PID, and gain-scheduled responses

clc; clear; close all;

%% Plant
K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

%% PID (same as pid_design)
[C, ~] = pidtune(G, 'PID');
T_pid = feedback(C * G, 1);

%% Gain Scheduled (approx using best high-performance gains)
C_gs = pid(8, 1, 0.02);   % representative aggressive gains
T_gs = feedback(C_gs * G, 1);

%% Plot comparison
figure;
hold on;
[y1, t1] = step(G);
[y2, t2] = step(T_pid);
[y3, t3] = step(T_gs);

figure;
plot(t1, y1, 'r--', 'LineWidth', 1.5); hold on;
plot(t2, y2, 'b-', 'LineWidth', 1.5);
plot(t3, y3, 'g-', 'LineWidth', 1.5);

grid on;
legend('Open-loop', 'PID', 'Gain-Scheduled');
title('Throttle–Motor Response Comparison');
xlabel('Time (seconds)');
ylabel('Motor Speed');
legend('Open-loop', 'PID', 'Gain-Scheduled');
title('Throttle–Motor Response Comparison');
xlabel('Time (seconds)');
ylabel('Motor Speed');
grid on;

% Save
saveas(gcf, 'comparison_plot.png');