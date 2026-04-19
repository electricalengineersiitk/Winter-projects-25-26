% pid_design.m
% PID design using MATLAB pidtune()

clc;
clear;
close all;

%% Plant definition
K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

%% Use pidtune to get controller
[C, info] = pidtune(G, 'PID');

% Extract gains
Kp = C.Kp;
Ki = C.Ki;
Kd = C.Kd;

fprintf('Tuned PID Gains:\n');
fprintf('Kp = %.3f\n', Kp);
fprintf('Ki = %.3f\n', Ki);
fprintf('Kd = %.3f\n', Kd);

%% Closed-loop system
T = feedback(C * G, 1);

%% Plot response
figure;
step(T);
title('Closed-Loop PID Response (pidtune)');
xlabel('Time (seconds)');
ylabel('Motor Speed');

%% Performance metrics
step_info = stepinfo(T);
disp('Step Response Info:');
disp(step_info);

% Save
saveas(gcf, 'pid_response.png');
