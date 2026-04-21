% GAIN SCHEDULED CONTROLLER FOR EV THROTTLE SYSTEM

% This script demonstrates gain scheduling for a throttle-controlled motor.
% Different PID gains are used depending on the throttle zone.

% Zones:
% Zone 1 : 0–30% throttle  -> smooth response (higher Ki)
% Zone 2 : 30–70% throttle -> balanced response
% Zone 3 : 70–100% throttle -> aggressive response (higher Kp, lower Ki)

% Throttle sequence simulated:
% 0–5 sec   -> 20% throttle
% 5–10 sec  -> 50% throttle
% 10–15 sec -> 85% throttle

clc;
clear;
close all;

%STEP 1 : DEFINE MOTOR PLANT MODEL
% Motor parameters
K = 1;          % steady state gain
tau = 0.5;      % time constant (seconds)

% Transfer function of the motor
G = tf(K,[tau 1]);

% STEP 2 : DEFINE PID GAINS FOR EACH THROTTLE ZONE

% Zone 1 (Low throttle 0–30%)
% Smooth response, higher Ki removes steady-state droop
zone1 = [2 3 0.05];    % [Kp Ki Kd]

% Zone 2 (Mid throttle 30–70%)
% Balanced controller gains
zone2 = [3 2 0.1];

% Zone 3 (High throttle 70–100%)
% Faster acceleration, higher Kp and lower Ki
zone3 = [4 1 0.15];

% STEP 3 : DEFINE THROTTLE SETPOINT SEQUENCE

% Simulation time segments
t1 = 0:0.01:5;        % Zone 1 duration
t2 = 5:0.01:10;       % Zone 2 duration
t3 = 10:0.01:15;      % Zone 3 duration

% Throttle levels
u1 = 0.20;            % 20% throttle
u2 = 0.50;            % 50% throttle
u3 = 0.85;            % 85% throttle

% STEP 4 : BUILD PID CONTROLLERS FOR EACH ZONE

% Zone 1 controller
C1 = pid(zone1(1),zone1(2),zone1(3));
T1 = feedback(C1*G,1);

% Zone 2 controller
C2 = pid(zone2(1),zone2(2),zone2(3));
T2 = feedback(C2*G,1);

% Zone 3 controller
C3 = pid(zone3(1),zone3(2),zone3(3));
T3 = feedback(C3*G,1);

% STEP 5 : SIMULATE MOTOR RESPONSE IN EACH ZONE

% Zone 1 simulation (0–5s)
[y1,t_out1] = step(T1,t1);
y1 = y1 * u1;        % scale according to throttle

% Zone 2 simulation (5–10s)
[y2,t_out2] = step(T2,t2-5);
y2 = y2 * u2;

% Zone 3 simulation (10–15s)
[y3,t_out3] = step(T3,t3-10);
y3 = y3 * u3;

% STEP 6 : COMBINE RESPONSES INTO ONE CONTINUOUS SIGNAL
time = [t_out1 ; t_out2+5 ; t_out3+10];
speed = [y1 ; y2 ; y3];

% STEP 7 : PLOT MOTOR SPEED RESPONSE

figure
plot(time,speed,'LineWidth',2)
hold on
grid on

title('Gain Scheduled EV Throttle Control')
xlabel('Time (seconds)')
ylabel('Motor Speed')

% STEP 8 : MARK ZONE TRANSITIONS

% Mark where the throttle zone changes occur
xline(5,'--r','Zone1 → Zone2');
xline(10,'--r','Zone2 → Zone3');

legend('Motor Speed','Zone Transition')
    mkdir('results');

% RESULT

saveas(gcf,'results/gainscheduled_response.png');

% DISPLAY ZONE GAINS

disp('PID Gains Used in Gain Scheduling:')
fprintf('\nZone 1 (0–30%% throttle):  Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone1)
fprintf('Zone 2 (30–70%% throttle): Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone2)
fprintf('Zone 3 (70–100%% throttle):Kp = %.2f  Ki = %.2f  Kd = %.2f\n',zone3)