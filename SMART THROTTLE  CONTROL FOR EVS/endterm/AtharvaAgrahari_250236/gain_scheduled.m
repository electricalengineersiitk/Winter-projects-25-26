% gain_scheduled.m
% 20% → 50% → 85% and plots the motor speed response.


%% Plant model

K = 1; tau = 0.5;
G = tf(K, [tau, 1]);

%% Design gains for each zone using pidtune

% Zone boundaries (in percent throttle)
zone_thresholds = [30, 70];   % 0-30% (Zone 1), 30-70% (Zone 2), 70-100% (Zone 3)

% Desired closed-loop bandwidths (rad/s) for each zone
bandwidths = [2, 5, 10];   % slow for low throttle, fast for high throttle

zone_gains = struct('Kp', [], 'Ki', [], 'Kd', []);
for zone = 1:3
    C = pidtune(G, 'pid', bandwidths(zone));
    zone_gains.Kp(zone) = C.Kp;
    zone_gains.Ki(zone) = C.Ki;
    zone_gains.Kd(zone) = C.Kd;
end

%% Simulation settings
dt = 0.01;          % time step (s)
t_end = 8;          % simulation duration (s)
time = 0:dt:t_end;

% Throttle setpoint sequence (percent)
setpoint = zeros(size(time));
for i = 1:length(time)
    if time(i) <= 2
        setpoint(i) = 20;          % Zone 1
    elseif time(i) <= 4
        setpoint(i) = 50;          % Zone 2
    else
        setpoint(i) = 85;          % Zone 3
    end
end
setpoint_norm = setpoint / 100;    % normalize to [0,1]

%% Simulation loop (Euler integration)

n = length(time);
y = zeros(1, n);          % motor speed (normalized)
u = zeros(1, n);          % control signal
integral_error = 0;
prev_error = 0;

for i = 1:n
    r = setpoint_norm(i);
    sp_pct = r * 100;
    
    % Determine current zone
    if sp_pct <= zone_thresholds(1)
        zone = 1;
    elseif sp_pct <= zone_thresholds(2)
        zone = 2;
    else
        zone = 3;
    end
    
    Kp = zone_gains.Kp(zone);
    Ki = zone_gains.Ki(zone);
    Kd = zone_gains.Kd(zone);
    
    error = r - y(i);
    P = Kp * error;
    I = Ki * integral_error;
    D = Kd * (error - prev_error) / dt;
    u(i) = P + I + D;
    u(i) = max(0, min(1, u(i)));   % saturate throttle to [0,1]
    
    % Anti-windup: freeze integrator when saturated
    if u(i) > 0 && u(i) < 1
        integral_error = integral_error + error * dt;
    end
    
    prev_error = error;
    
    % Plant update: dy/dt = (K*u - y)/tau
    if i < n
        dy = (K * u(i) - y(i)) / tau * dt;
        y(i+1) = y(i) + dy;
    end
end

y_pct = y * 100;

%% Plot response

figure;
plot(time, setpoint, 'k--', 'LineWidth', 1.5); hold on;
plot(time, y_pct, 'b-', 'LineWidth', 1.5);
xlabel('Time (seconds)');
ylabel('Throttle / Motor Speed (%)');
title('Gain-Scheduled Throttle Control Response');
grid on;
legend('Setpoint (Throttle Command)', 'Motor Speed', 'Location', 'best');
xline(2, 'r--', 'Zone 1→2', 'LabelVerticalAlignment', 'top');
xline(4, 'r--', 'Zone 2→3', 'LabelVerticalAlignment', 'top');

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/gainscheduled_response.png');

%% Display gains

fprintf('Gain-scheduled controller\n');
fprintf('Zone boundaries: 0-30%%, 30-70%%, 70-100%%\n');
fprintf('Zone gains (designed with pidtune):\n');
fprintf('  Zone 1 (low):  Kp = %.2f, Ki = %.2f, Kd = %.2f\n', ...
    zone_gains.Kp(1), zone_gains.Ki(1), zone_gains.Kd(1));
fprintf('  Zone 2 (mid):  Kp = %.2f, Ki = %.2f, Kd = %.2f\n', ...
    zone_gains.Kp(2), zone_gains.Ki(2), zone_gains.Kd(2));
fprintf('  Zone 3 (high): Kp = %.2f, Ki = %.2f, Kd = %.2f\n', ...
    zone_gains.Kp(3), zone_gains.Ki(3), zone_gains.Kd(3));