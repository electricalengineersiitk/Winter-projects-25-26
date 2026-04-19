clc;
clear;
close all;

% Motor model

K = 1;
tau = 0.5;

t = 0:0.01:15;
dt = t(2) - t(1);

% Throttle profile

u = zeros(size(t));

for i = 1:length(t)
    if t(i) < 5
        u(i) = 0.20;
    elseif t(i) < 10
        u(i) = 0.50;
    else
        u(i) = 0.85;
    end
end

% PID gains for zones

zone1 = [2 3 0.05];
zone2 = [3 2 0.1];
zone3 = [4 1 0.15];

% Initialize system

x = 0;                  % motor speed
integral_error = 0;
prev_error = 0;

y = zeros(size(t));

for i = 1:length(t)
    
    % Select gains based on throttle
    
    if u(i) <= 0.3
        Kp = zone1(1); Ki = zone1(2); Kd = zone1(3);
    elseif u(i) <= 0.7
        Kp = zone2(1); Ki = zone2(2); Kd = zone2(3);
    else
        Kp = zone3(1); Ki = zone3(2); Kd = zone3(3);
    end
    
    % Error
    error = u(i) - x;
    
    % Integral
    integral_error = integral_error + error*dt;
    
    % Derivative
    derivative = (error - prev_error)/dt;
    
    % PID control
    control = Kp*error + Ki*integral_error + Kd*derivative;
    
    % Motor dynamics
    dx = (-x + control)/tau;
    
    % State update
    x = x + dx*dt;
    
    % Store output
    y(i) = x;
    
    prev_error = error;
end

% Plot

figure;
plot(t,y,'b','LineWidth',2)
hold on
grid on

title('Gain Scheduled EV Throttle Control (Correct)');
xlabel('Time (seconds)');
ylabel('Motor Speed');

% Zone transitions

xline(5,'--r','Zone1 → Zone2');
xline(10,'--r','Zone2 → Zone3');

legend('Motor Speed','Zone Transition');

% Save

if ~exist('results','dir')
    mkdir('results');
end

saveas(gcf,'results/gainscheduled_response_correct.png');

% Display gains

disp('PID Gains Used in Gain Scheduling:')
fprintf('\nZone 1:  Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone1)
fprintf('Zone 2:  Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone2)
fprintf('Zone 3:  Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone3)