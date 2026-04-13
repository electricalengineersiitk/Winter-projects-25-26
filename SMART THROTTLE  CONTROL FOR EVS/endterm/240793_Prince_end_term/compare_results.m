clc; clear; close all;

% Plant
K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

% PID
C_pid = pid(3,5,0.1);
T_pid = feedback(C_pid*G,1);

% Gain Scheduled (use best zone approximation)
C_gs = pid(5,2,0.1);
T_gs = feedback(C_gs*G,1);

figure;
hold on;

% Open-loop
step(G, 'r--');

% PID
step(T_pid, 'b');

% Gain-scheduled
step(T_gs, 'g');

legend('Open-loop','PID','Gain-Scheduled');
title('Throttle–Motor Response Comparison');
xlabel('Time (s)');
ylabel('Speed');

saveas(gcf, 'comparison_plot.png');