% pid_design.m
% targets: rise time < 1 s, overshoot < 10%, steady-state error < 2%.


% Plant model
K = 1; tau = 0.5;
G = tf(K, [tau, 1]);

% Manually tuned PID gains
% Kp = 2.5;
% Ki = 1.2;
% Kd = 0.15;
% C = pid(Kp, Ki, Kd);
C = pidtune(G, 'PID');

% Closed-loop transfer function (unity feedback)
T = feedback(C * G, 1);

% Step response
figure;
step(T);
title('Closed-Loop PID Step Response');
xlabel('Time (seconds)');
ylabel('Motor Speed (normalized)');
grid on;

% Evaluate performance
info = stepinfo(T);
rise_time = info.RiseTime;
overshoot = info.Overshoot;
steady_state_error = abs(1 - dcgain(T)) * 100;  % in percent

fprintf('PID Gains: Kp = %.2f, Ki = %.2f, Kd = %.2f\n', Kp, Ki, Kd);
fprintf('Performance:\n');
fprintf('  Rise time: %.2f s (target < 1 s)  \n', rise_time)
fprintf('  Overshoot: %.1f%% (target < 10%%)  \n', overshoot)
fprintf('  Steady-state error: %.2f%% (target < 2%%)  \n', steady_state_error);

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/pid_response.png');
