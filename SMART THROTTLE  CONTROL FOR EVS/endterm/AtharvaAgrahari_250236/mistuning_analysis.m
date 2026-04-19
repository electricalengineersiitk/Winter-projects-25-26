%% mistuning_analysis.m
% Demonstrate the effect of deliberately mistuning the PI controller.
% Compare:
%   1. Correctly tuned PI (Kp=4.7, Ki=19.79)
%   2. Too high Kp (Kp=10, Ki=19.79)   → increases damping, reduces overshoot
%   3. Too high Ki (Kp=4.7, Ki=50)     → reduces damping, increases overshoot


% Plant model (first-order)
K_plant = 1; tau = 0.5;
G = tf(K_plant, [tau 1]);

% Correct gains (from pidTuner)
Kp_correct = 4.7;
Ki_correct = 19.79;
Kd = 0;

% Mistuned gains
Kp_high = 10;        % higher proportional gain
Ki_high = 50;        % higher integral gain

% Closed-loop systems
C_correct = pid(Kp_correct, Ki_correct, Kd);
T_correct = feedback(C_correct * G, 1);

C_highKp = pid(Kp_high, Ki_correct, Kd);
T_highKp = feedback(C_highKp * G, 1);
info1 = stepinfo(T_highKp);
fprintf("Overshoot (Kp = 10) : %.2f %%\n", info1.Overshoot);

C_highKi = pid(Kp_correct, Ki_high, Kd);
T_highKi = feedback(C_highKi * G, 1);
info2 = stepinfo(T_highKi);
fprintf("Overshoot (Ki = 50) : %.2f %%\n", info2.Overshoot);

% Step response comparison
figure;
step(T_correct, 'b-', T_highKp, 'r--', T_highKi, 'g-.', 2);
legend('Correctly tuned', 'Too high Kp', 'Too high Ki', 'Location', 'best');
title('Effect of PI Mistuning on Step Response');
xlabel('Time (s)');
ylabel('Motor Speed (rpm)');
grid on;

% Annotate with correct observations
text(0.8, 0.9, 'High Kp → more damping, less overshoot', 'Color', 'r');
text(0.8, 0.7, 'High Ki → less damping, more overshoot', 'Color', 'g');

% Save figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/mistuning_analysis.png');