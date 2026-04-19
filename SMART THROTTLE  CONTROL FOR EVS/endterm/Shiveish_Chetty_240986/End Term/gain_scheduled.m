% gain_scheduled.m
% 3-zone gain scheduling with smooth boundary transitions

clc;
clear;
close all;

%% Plant
K = 1;
tau = 0.5;

%% Time
t = 0:0.01:20;
dt = t(2) - t(1);

%% Throttle profile
u = zeros(size(t));
for i = 1:length(t)
    if t(i) < 6
        u(i) = 0.2;
    elseif t(i) < 12
        u(i) = 0.5;
    else
        u(i) = 0.85;
    end
end

%% Define 3 CLEAR PID zones
zone1 = [3 3 0.3];    % smooth
zone2 = [6 2 0.1];    % balanced
zone3 = [8 1 0.02];   % aggressive

blend_width = 0.05;   % small smoothing region

%% Init
y = zeros(size(t));
e_prev = 0;
integral = 0;

%% Simulation
for i = 2:length(t)
    
    % -------- Zone selection with smoothing --------
    if u(i) < (0.3 - blend_width)
        gains = zone1;
        
    elseif u(i) < (0.3 + blend_width)
        % Blend zone1 → zone2
        alpha = (u(i) - (0.3 - blend_width)) / (2*blend_width);
        gains = (1-alpha)*zone1 + alpha*zone2;
        
    elseif u(i) < (0.7 - blend_width)
        gains = zone2;
        
    elseif u(i) < (0.7 + blend_width)
        % Blend zone2 → zone3
        alpha = (u(i) - (0.7 - blend_width)) / (2*blend_width);
        gains = (1-alpha)*zone2 + alpha*zone3;
        
    else
        gains = zone3;
    end
    
    Kp = gains(1);
    Ki = gains(2);
    Kd = gains(3);

    % -------- PID --------
    e = u(i) - y(i-1);
    integral = integral + e * dt;
    derivative = (e - e_prev) / dt;

    control = Kp*e + Ki*integral + Kd*derivative;

    % -------- Plant --------
    dy = (-y(i-1) + control) / tau;
    y(i) = y(i-1) + dy * dt;

    e_prev = e;
end

%% Plot
figure;
plot(t, y, 'LineWidth', 2);
hold on;
xline(6, '--r', 'Zone 1 → 2');
xline(12, '--r', 'Zone 2 → 3');

title('Gain-Scheduled PID (3 Clear Zones + Smooth Transition)');
xlabel('Time (seconds)');
ylabel('Motor Speed');
grid on;
% Save
saveas(gcf, 'gainscheduled_response.png');