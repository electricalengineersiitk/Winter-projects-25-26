clc; clear; close all;

% Motor model (same as before)
K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

% Initial PID values
Kp = 3;
Ki = 5;
Kd = 0.1;

% Create PID controller
C = pid(Kp, Ki, Kd);

% Closed-loop system
T = feedback(C * G, 1);

% Plot response
figure;
step(T);
title('PID Controlled Response');
xlabel('Time (s)');
ylabel('Speed');

% Performance info
info = stepinfo(T)

% Save plot
saveas(gcf, 'pid_response.png');