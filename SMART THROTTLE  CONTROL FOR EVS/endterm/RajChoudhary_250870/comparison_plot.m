%% COMPARISON : Open-Loop  vs  Single PID  vs  Gain-Scheduled PID
%  Plant : G(s) = K / (tau*s + 1)  ,  K=1 , tau=0.1
%  Same throttle step profile used across all three methods

clear;
clc;
close all;

%% Common Plant Parameters
K   = 1;
tau = 0.1;
G   = tf(K, [tau 1]);

%% Time & Throttle Profile
dt = 0.001;
t  = 0:dt:10;

throttle = zeros(size(t));
throttle(t < 3)           = 0.2;     % Zone 1
throttle(t >= 3 & t < 7) = 0.5;     % Zone 2
throttle(t >= 7)          = 0.85;    % Zone 3

N = length(t);

%  METHOD 1 : Open-Loop  (no controller, input = throttle directly)
y_ol   = zeros(1, N);

for i = 2:N
    u_ol    = throttle(i);           % raw throttle fed directly to plant
    y_ol(i) = y_ol(i-1) + dt * ((-y_ol(i-1) + K*u_ol) / tau);
end


%  METHOD 2 : Single Fixed PID  (pidtune values from pid_design.m)
%  C = pidtune(G,'PID') for K=1, tau=0.1 gives approx:
%  Kp = 1.43 , Ki = 25.9 , Kd = 0  (auto-tuned baseline)

Kp_f = 1.43;
Ki_f = 25.9;
Kd_f = 0;

y_pid    = zeros(1, N);
int_pid  = 0;
eprev_pid = 0;

for i = 2:N
    e_pid     = throttle(i) - y_pid(i-1);
    int_pid   = int_pid + e_pid * dt;
    deriv_pid = (e_pid - eprev_pid) / dt;
    u_pid     = Kp_f*e_pid + Ki_f*int_pid + Kd_f*deriv_pid;
    u_pid     = max(0, min(1, u_pid));
    y_pid(i)  = y_pid(i-1) + dt * ((-y_pid(i-1) + K*u_pid) / tau);
    eprev_pid = e_pid;
end

%  METHOD 3 : Gain-Scheduled PID  (from gain_scheduled.m)
Zone(1).Kp = 2.;  Zone(1).Ki = 1.5;  Zone(1).Kd = 0.01;
Zone(2).Kp = 3.0;  Zone(2).Ki = 2.5;  Zone(2).Kd = 0.02;
Zone(3).Kp = 4.5;  Zone(3).Ki = 4;  Zone(3).Kd = 0.03;

y_gs    = zeros(1, N);
int_gs  = 0;
eprev_gs = 0;
z_prev  = 1;
zone_log = zeros(1, N);

for i = 2:N
    e_gs = throttle(i) - y_gs(i-1);

    if throttle(i) < 0.3
        z = 1;
    elseif throttle(i) < 0.7
        z = 2;
    else
        z = 3;
    end

    if z ~= z_prev
        int_gs = 0;
    end
    z_prev = z;

    Kp = Zone(z).Kp;  Ki = Zone(z).Ki;  Kd = Zone(z).Kd;
    zone_log(i) = z;

    int_gs    = int_gs + e_gs * dt;
    deriv_gs  = (e_gs - eprev_gs) / dt;
    u_gs      = Kp*e_gs + Ki*int_gs + Kd*deriv_gs;
    u_gs      = max(0, min(1, u_gs));
    y_gs(i)   = y_gs(i-1) + dt * ((-y_gs(i-1) + K*u_gs) / tau);
    eprev_gs  = e_gs;
end

%  PERFORMANCE METRICS
methods     = {'Open-Loop', 'Fixed PID', 'Gain-Scheduled'};
y_all       = {y_ol, y_pid, y_gs};
zone_starts = [1, find(t>=3,1), find(t>=7,1)];
zone_ends   = [find(t>=3,1)-1, find(t>=7,1)-1, N];
sp_vals     = [0.2, 0.5, 0.85];

fprintf('PERFORMANCE COMPARISON  (K=1, tau=0.1)\n');


for m = 1:3
    y_m = y_all{m};
    fprintf('\n--- %s ---\n', methods{m});
    fprintf('Zone | Setpoint | Final Output | SS Error\n');
    for z = 1:3
        y_end = y_m(zone_ends(z));
        ss_err = abs(sp_vals(z) - y_end) / sp_vals(z) * 100;
        fprintf('  %d  |   %.2f   |    %.4f    |  %.2f%%\n', z, sp_vals(z), y_end, ss_err);
    end
end



%  PLOTTING

figure;

% Plot 1 : All three responses on one axes
plot(t, y_ol,  'y',  'LineWidth', 1.5); hold on;
plot(t, y_pid, 'g',  'LineWidth', 1.5);
plot(t, y_gs,  'b',  'LineWidth', 2);
plot(t, throttle, 'w--', 'LineWidth', 1.2);
xline(3, '--r', 'Zone 1→2', 'LabelVerticalAlignment','bottom');
xline(7, '--r', 'Zone 2→3', 'LabelVerticalAlignment','bottom');
hold off;
grid on;
ylim([0 1.2]);
title('Response Comparison : Open-Loop vs Fixed PID vs Gain-Scheduled');
xlabel('Time (s)');
ylabel('Motor Speed');
legend('Open-Loop','Fixed PID','Gain-Scheduled','Setpoint','Location','southeast');


