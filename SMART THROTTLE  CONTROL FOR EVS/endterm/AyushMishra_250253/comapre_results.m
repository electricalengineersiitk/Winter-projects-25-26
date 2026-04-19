clc;
clear;
close all;

% Motor model

K = 1;
tau = 0.5;
G = tf(K,[tau 1]);

t = 0:0.01:10;
dt = t(2) - t(1);

% Open loop response

[y_ol,t] = step(G,t);

% PID controller (closed loop)

C_pid = pid(3,2,0.1);
T_pid = feedback(C_pid*G,1);
[y_pid,~] = step(T_pid,t);

% Gain scheduling (dynamic simulation)

y_gs = zeros(size(t));
x = 0;              % system state
integral_error = 0; % for integral action
ref = 1;            % step input

for i = 1:length(t)
    
    % Select controller gains based on zone
    if t(i) < 3
        Kp = 2; Ki = 1;
    elseif t(i) < 6
        Kp = 3; Ki = 2;
    else
        Kp = 4; Ki = 3;
    end
    
    % Error
    error = ref - x;
    
    % Integral update
    integral_error = integral_error + error*dt;
    
    % Control input (PI control)
    u = Kp*error + Ki*integral_error;
    
    % Plant dynamics: dx/dt = (-x + u)/tau
    dx = (-x + u)/tau;
    
    % State update (Euler integration)
    x = x + dx*dt;
    
    % Output
    y_gs(i) = x;
end

% Performance metrics

info_pid = stepinfo(T_pid);

% Plot

figure;
hold on;
grid on;
box on;

% Shaded zones

yl = [0 1.5];

fill([0 3 3 0],[yl(1) yl(1) yl(2) yl(2)], ...
    [0.9 0.9 1],'EdgeColor','none','FaceAlpha',0.3);

fill([3 6 6 3],[yl(1) yl(1) yl(2) yl(2)], ...
    [0.9 1 0.9],'EdgeColor','none','FaceAlpha',0.3);

fill([6 10 10 6],[yl(1) yl(1) yl(2) yl(2)], ...
    [1 0.9 0.9],'EdgeColor','none','FaceAlpha',0.3);

% Plot responses

h1 = plot(t,y_ol,'r--','LineWidth',2);
h2 = plot(t,y_pid,'b-','LineWidth',2);
h3 = plot(t,y_gs,'g-.','LineWidth',2);

% Zone boundaries

xline(3,'k--','LineWidth',1.5);
xline(6,'k--','LineWidth',1.5);

% Zone labels

text(1.5,1.35,'Zone 1','HorizontalAlignment','center');
text(4.5,1.35,'Zone 2','HorizontalAlignment','center');
text(8,1.35,'Zone 3','HorizontalAlignment','center');

% Rise time marker (PID)

rt = info_pid.RiseTime;
y_rt = interp1(t,y_pid,rt);

plot(rt,y_rt,'bo','MarkerFaceColor','b');
text(rt,y_rt, sprintf('  Rise = %.2fs',rt));

% Peak and overshoot marker (PID)

[peak_val, idx] = max(y_pid);
peak_time = t(idx);

plot(peak_time,peak_val,'ko','MarkerFaceColor','k');

text(peak_time,peak_val, ...
    sprintf('  Peak = %.2f\n  OS = %.2f%%', ...
    peak_val, info_pid.Overshoot));

% Labels

title('Throttle Motor Response Comparison');
xlabel('Time (seconds)');
ylabel('Motor Speed');

legend([h1 h2 h3], ...
    'Open Loop','PID','Gain Scheduled', ...
    'Location','best');

% Save figure

if ~exist('results','dir')
    mkdir('results');
end

exportgraphics(gcf,'results/comparison_plot_final.png','Resolution',300);

% Display performance

disp("Performance Summary:");

fprintf('\nPID:\n');
fprintf('Rise Time: %.3f s\n', info_pid.RiseTime);
fprintf('Settling Time: %.3f s\n', info_pid.SettlingTime);
fprintf('Overshoot: %.2f %%\n', info_pid.Overshoot);