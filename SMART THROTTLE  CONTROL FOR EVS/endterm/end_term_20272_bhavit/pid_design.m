% % pid_design.m — PID Controller Design and Tuning % EV Throttle Control |
    End - Term Project |
    Bhavit Meena(240272) %
        % Purpose : % Designs a PID controller in series with the motor plant,
    simulates the % closed - loop step response,
    and reports performance metrics.%
        % Controller Equation : % U(s) = Kp * E(s) + Ki / s * E(s) +
                                         Kd * s *
                                             E(s)

                                                 clear;
clc;
close all;

% % -- -Plant(from plant_model.m)-- - K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

% % -- -PID Gains(tuned manually)-- - % Tuning strategy : %
                                                          1. Start
    : Kp = 2,
      Ki = 0, Kd = 0 — get rough speed,
      check overshoot %
          2. Add Ki = 1 — eliminates steady - state error(SSE) % 3. Add Kd =
                          0.1 — damps the slight overshoot introduced by Ki
                          % Result : fast rise,
      < 5 % overshoot, zero SSE Kp = 3.0;
Ki = 5.0;
Kd = 0.2;

C = pid(Kp, Ki, Kd);

% % -- -Closed - Loop System-- - T_pid = feedback(C * G, 1);
% Unity negative feedback fprintf('PID Transfer Function:\n');
disp(C);
fprintf('Closed-Loop Transfer Function:\n');
disp(T_pid);

% % -- -Step Response-- - t = 0 : 0.01 : 5;
figure('Name', 'PID Closed-Loop Step Response', 'NumberTitle', 'off');
[ y, t_out ] = step(T_pid, t);
plot(t_out, y, 'b-', 'LineWidth', 2);
yline(1.0, 'k--', 'Setpoint', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Motor Speed (normalised)');
title('PID Closed-Loop Step Response');
legend(sprintf('PID (Kp=%.1f, Ki=%.1f, Kd=%.1f)', Kp, Ki, Kd),
       ... 'Setpoint = 1.0', 'Location', 'southeast');

% % -- -Performance Metrics-- - info = stepinfo(T_pid);
sse = abs(1 - dcgain(T_pid));

fprintf('\n--- PID Closed-Loop Performance ---\n');
fprintf('  Kp = %.2f   Ki = %.2f   Kd = %.2f\n', Kp, Ki, Kd);
fprintf('  Rise Time      : %.4f s\n', info.RiseTime);
fprintf('  Settling Time  : %.4f s\n', info.SettlingTime);
fprintf('  Overshoot      : %.2f %%\n', info.Overshoot);
fprintf('  Steady-State Error : %.6f\n', sse);

% % -- -Save-- - saveas(gcf, fullfile(fileparts(mfilename('fullpath')),
                                      'results', 'pid_response.png'));
fprintf('\nSaved: results/pid_response.png\n');
