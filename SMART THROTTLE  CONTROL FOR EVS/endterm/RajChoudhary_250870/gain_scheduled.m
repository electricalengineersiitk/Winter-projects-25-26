
%% Plant — K = 1 , tau = 0.1 
K   = 1;
tau = 0.1;
G   = tf(K, [tau 1]);

%% PID Gains per Zone — re-tuned for tau = 0.1

% Zone 1 : throttle < 0.3  (low speed)
% Zone 2 : throttle < 0.7  (mid speed)
% Zone 3 : throttle >= 0.7 (high speed)

Zone(1).Kp = 2.;  Zone(1).Ki = 1.5;  Zone(1).Kd = 0.01;
Zone(2).Kp = 3.0;  Zone(2).Ki = 2.5;  Zone(2).Kd = 0.02;
Zone(3).Kp = 4.5;  Zone(3).Ki = 4;  Zone(3).Kd = 0.03;

fprintf('  1  | < 0.30         |  %.2f   %.2f   %.3f\n', Zone(1).Kp, Zone(1).Ki, Zone(1).Kd);
fprintf('  2  | 0.30 – 0.70    |  %.2f   %.2f   %.3f\n', Zone(2).Kp, Zone(2).Ki, Zone(2).Kd);
fprintf('  3  | >= 0.70        |  %.2f   %.2f   %.3f\n', Zone(3).Kp, Zone(3).Ki, Zone(3).Kd);

%% Time settings
% tau=0.1 plant settles ~5*tau = 0.5s per zone, so t_end=10s gives
% enough room for all three zones to fully settle
dt = 0.001;
t  = 0:dt:10;

%% Throttle profile (step sequence)
throttle = zeros(size(t));
throttle(t < 3)              = 0.2;    % Zone 1
throttle(t >= 3 & t < 7)    = 0.5;    % Zone 2
throttle(t >= 7)             = 0.85;   % Zone 3

%% Initialize variables
y        = zeros(size(t));
e_prev   = 0;
integral = 0;
zone_log = zeros(size(t));
z_prev   = 1;

%% Simulation loop
for i = 2:length(t)

    % Current error
    e = throttle(i) - y(i-1);

    % -------- Gain Selection (ZONE SWITCHING) --------
    if throttle(i) < 0.3
        z = 1;
    elseif throttle(i) < 0.7
        z = 2;
    else
        z = 3;
    end

    % Reset integrator on zone switch to prevent wind-up carry-over
    if z ~= z_prev
        integral = 0;
    end
    z_prev = z;

    Kp = Zone(z).Kp;
    Ki = Zone(z).Ki;
    Kd = Zone(z).Kd;
    zone_log(i) = z;

    % -------- PID Control --------
    integral   = integral + e * dt;
    derivative = (e - e_prev) / dt;
    u_control  = Kp*e + Ki*integral + Kd*derivative;

    % Anti-windup : clamp to [0, 1] matching setpoint scale
    u_control  = max(0, min(1, u_control));

    % -------- Plant update (First-order Euler) --------
    y(i) = y(i-1) + dt * ((-y(i-1) + K*u_control) / tau);

    e_prev = e;
end

%% Plot
figure;

% Plot 1 : Motor response vs setpoint
plot(t, y,'b',  'LineWidth', 2);
hold on;
plot(t, throttle, 'w--','LineWidth', 1.2);
xline(3, '--r', 'Zone 1 \rightarrow 2', 'LabelVerticalAlignment', 'bottom');
xline(7, '--r', 'Zone 2 \rightarrow 3', 'LabelVerticalAlignment', 'bottom');
hold off;
grid on;
ylim([0 1.2]);
title('Gain-Scheduled Motor Response  (tau = 0.1)');
xlabel('Time (s)');
ylabel('Motor Speed (normalised)');
legend('Output', 'Setpoint', 'Location', 'southeast');

