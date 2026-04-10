% plant_model.m

% Motor parameters
K = 1;          % steady-state gain (normalized)
tau = 0.5;      % time constant (seconds)

% Transfer function G(s) = K / (tau*s + 1)
G = tf(K, [tau, 1]);

% Open-loop step response (input = 1 => 100% throttle)
figure;
step(G);
title('Open-Loop Motor Step Response');
xlabel('Time');
ylabel('Motor Speed (normalized)');
grid on;

% Save the figure
if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, 'results/open_loop_response.png');

% Display step response characteristics
info = stepinfo(G);
fprintf('Open-loop step response:\n');
fprintf('  Rise time: %.2f s\n', info.RiseTime);
fprintf('  Settling time: %.2f s\n', info.SettlingTime);