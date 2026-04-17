% compare_results.m

%% Plant model

K = 1; tau = 0.5;
G = tf(K, [tau, 1]);

%% Fixed PID controller (from pid_design.m)

Kp_fixed = 4.70; Ki_fixed = 19.79; Kd_fixed = 0.00;
C_fixed = pid(Kp_fixed, Ki_fixed, Kd_fixed);
T_fixed = feedback(C_fixed * G, 1);

%% Gain-scheduled controller gains for high throttle (Zone 3)

% These are the same as in gain_scheduled.m for zone 3
Kp_sched = 4.7;
Ki_sched = 19.79;
Kd_sched = 0.00;

%% Simulation parameters for gain-scheduled (Euler integration)

setpoint = 0.85; % 85% throttle step
dt = 0.01;
t = 0:dt:5;
n = length(t);
y_sched = zeros(1, n);
u_sched = zeros(1, n);
integral_error = 0;
prev_error = 0;

for i = 1:n
    error = setpoint - y_sched(i);
    P = Kp_sched * error;
    I = Ki_sched * integral_error;
    D = Kd_sched * (error - prev_error) / dt;
    u_sched(i) = P + I + D;
    u_sched(i) = max(0, min(1, u_sched(i)));
    if u_sched(i) > 0 && u_sched(i) < 1
        integral_error = integral_error + error * dt;
    end
    prev_error = error;
    if i < n
        dy = (K * u_sched(i) - y_sched(i)) / tau * dt;
        y_sched(i+1) = y_sched(i) + dy;
    end
end

%% Open-loop and fixed PID responses

[y_open, t_open] = step(setpoint * G, t);
[y_pid, t_pid] = step(setpoint * T_fixed, t);

%% Plot comparison

figure;
plot(t_open, y_open, 'r--', 'LineWidth', 1.5); hold on;
plot(t_pid, y_pid, 'b-', 'LineWidth', 1.5);
plot(t, y_sched, 'g-', 'LineWidth', 1.5);
xlabel('Time (seconds)');
ylabel('Motor Speed (normalized)');
title('Response Comparison for 85% Throttle Step');
grid on;
legend('Open-Loop', 'PID', 'Gain-Scheduled', 'Location', 'southeast');
yline(setpoint, 'k:', 'Setpoint');

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/comparison_plot.png');

%% Print performance metrics

info_pid = stepinfo(y_pid, t_pid, setpoint);
info_sched = stepinfo(y_sched, t, setpoint);
fprintf('Comparison for %d%% step:\n', round(setpoint*100));
fprintf('  Fixed PID:    Rise time = %.2f s, Overshoot = %.1f%%\n', ...
    info_pid.RiseTime, info_pid.Overshoot);
fprintf('  Gain-Scheduled: Rise time = %.2f s, Overshoot = %.1f%%\n', ...
    info_sched.RiseTime, info_sched.Overshoot);