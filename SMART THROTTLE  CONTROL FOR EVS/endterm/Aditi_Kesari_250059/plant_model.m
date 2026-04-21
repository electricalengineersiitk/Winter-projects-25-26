clc;
clear;
close all;

% Motor parameters
K = 1;          % steady state gain
tau = 0.5;      % time constant (seconds)

% Transfer function
G = tf(K,[tau 1]);

% Step response
figure;
step(G);

title('Open Loop Motor Step Response');
xlabel('Time (seconds)');
ylabel('Motor Speed');

grid on;

% Create results folder if not present
if ~exist('results','dir')
    mkdir('results');
end

% Save figure
saveas(gcf,'results/open_loop_response.png');

% Performance info
info = stepinfo(G)