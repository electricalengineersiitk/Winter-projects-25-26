clc;            % Clear command window
clear;          % Clear workspace variables
close all;      % Close all figures

% -------------------------------------------------
% STEP 1 : DEFINE MOTOR (PLANT) MODEL
% -------------------------------------------------

K = 1;          % Steady-state gain
tau = 0.5;      % Time constant (seconds)

% Transfer function:
% G(s) = K / (tau*s + 1)
G = tf(K,[tau 1]);

disp("Motor Transfer Function:")
G

% -------------------------------------------------
% STEP 2 : OPEN-LOOP STEP RESPONSE
% -------------------------------------------------

figure;
step(G);    % Open-loop response (no controller)

title('Open Loop Motor Step Response');
xlabel('Time (seconds)');
ylabel('Motor Speed');
grid on;

% Create results folder if not present
if ~exist('results','dir')
    mkdir('results');
end

% Save open-loop figure
saveas(gcf,'results/open_loop_response.png');

% Performance metrics for open-loop
info_open = stepinfo(G);

disp("Open Loop Performance:")
info_open

% -------------------------------------------------
% STEP 3 : DEFINE PID CONTROLLER
% -------------------------------------------------

% Tuned PID gains (can be adjusted)
Kp = 3;         % Proportional gain → faster response
Ki = 2;         % Integral gain → removes steady-state error
Kd = 0.1;       % Derivative gain → reduces overshoot

% Create PID controller
C = pid(Kp,Ki,Kd);

% -------------------------------------------------
% STEP 4 : CLOSED-LOOP SYSTEM
% -------------------------------------------------

% Closed-loop transfer function:
% T(s) = C(s)G(s) / (1 + C(s)G(s))
T = feedback(C*G,1);

% -------------------------------------------------
% STEP 5 : CLOSED-LOOP STEP RESPONSE
% -------------------------------------------------

figure;
step(T);    % Closed-loop response

title('Closed Loop Motor Step Response (PID Controlled)');
xlabel('Time (seconds)');
ylabel('Motor Speed');
grid on;

% Save closed-loop figure
saveas(gcf,'results/closed_loop_response.png');

% Performance metrics for closed-loop
info_closed = stepinfo(T);

disp("Closed Loop Performance:")
info_closed

% -------------------------------------------------
% STEP 6 : DISPLAY PID GAINS USED
% -------------------------------------------------

fprintf('\nPID Gains Used:\n')
fprintf('Kp = %.2f\n',Kp)
fprintf('Ki = %.2f\n',Ki)
fprintf('Kd = %.2f\n',Kd)

% -------------------------------------------------
% STEP 7 : COMPARISON PLOT (IMPORTANT)
% -------------------------------------------------

figure;
step(G, T);   % Plot open-loop and closed-loop together

title('Open Loop vs Closed Loop Comparison');
xlabel('Time (seconds)');
ylabel('Motor Speed');
legend('Open Loop','Closed Loop');
grid on;

% Save comparison figure
saveas(gcf,'results/comparison_plot.png');

% -------------------------------------------------
% STEP 8 : PRINT PERFORMANCE SUMMARY (CLEAN FORMAT)
% -------------------------------------------------

disp("===== PERFORMANCE COMPARISON =====")

fprintf('\n--- Open Loop ---\n');
fprintf('Rise Time: %.3f s\n', info_open.RiseTime);
fprintf('Settling Time: %.3f s\n', info_open.SettlingTime);
fprintf('Overshoot: %.2f %%\n', info_open.Overshoot);

fprintf('\n--- Closed Loop ---\n');
fprintf('Rise Time: %.3f s\n', info_closed.RiseTime);
fprintf('Settling Time: %.3f s\n', info_closed.SettlingTime);
fprintf('Overshoot: %.2f %%\n', info_closed.Overshoot);