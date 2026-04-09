clc; clear; close all;

% Define motor parameters
K = 1;
tau = 0.5;

% Transfer function
G = tf(K, [tau 1]);

% Plot step response
figure;
step(G);
title('Open-Loop Motor Step Response');
xlabel('Time (s)');
ylabel('Speed');

% Save plot
saveas(gcf, 'open_loop_response.png');