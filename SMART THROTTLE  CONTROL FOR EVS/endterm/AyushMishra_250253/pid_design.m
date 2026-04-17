clc;
clear;
close all;

% Motor (Plant) Definition

K = 1;
tau = 0.5;

G = tf(K,[tau 1]);

disp("Motor Transfer Function:")
G

% Create results folder
if ~exist('results','dir')
    mkdir('results');
end


%% Closed Loop: Kp Variation

Kp_values = [0.5 1 2 4 6];
Kp_results = [];

figure; hold on; grid on;

for i = 1:length(Kp_values)
    
    C = pid(Kp_values(i),0,0);
    T = feedback(C*G,1);

    step(T);

    info = stepinfo(T);

    Kp_results(i,:) = [Kp_values(i), info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime];

    fprintf('\nKp = %.2f\n', Kp_values(i));
    fprintf('Rise Time: %.3f | Settling: %.3f | Overshoot: %.2f | Peak: %.3f | PeakTime: %.3f\n', ...
        info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime);
end

title('Closed Loop: Kp Variation');
legend('0.5','1','2','4','6');
saveas(gcf,'results/Kp_variation.png');


%% Closed Loop: Ki Variation

Kp = 2;
Ki_values = [0.5 1 2 4 6];
Ki_results = [];

figure; hold on; grid on;

for i = 1:length(Ki_values)

    C = pid(Kp,Ki_values(i),0);
    T = feedback(C*G,1);

    step(T);

    info = stepinfo(T);

    Ki_results(i,:) = [Ki_values(i), info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime];

    fprintf('\nKi = %.2f\n', Ki_values(i));
    fprintf('Rise Time: %.3f | Settling: %.3f | Overshoot: %.2f | Peak: %.3f | PeakTime: %.3f\n', ...
        info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime);
end

title('Closed Loop: Ki Variation');
legend('0.5','1','2','4','6');
saveas(gcf,'results/Ki_variation.png');


%% Closed Loop: Kd Variation

Kp = 2; Ki = 2;
Kd_values = [0.01 0.05 0.1 0.2 0.5];
Kd_results = [];

figure; hold on; grid on;

for i = 1:length(Kd_values)

    C = pid(Kp,Ki,Kd_values(i));
    T = feedback(C*G,1);

    step(T);

    info = stepinfo(T);

    Kd_results(i,:) = [Kd_values(i), info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime];

    fprintf('\nKd = %.2f\n', Kd_values(i));
    fprintf('Rise Time: %.3f | Settling: %.3f | Overshoot: %.2f | Peak: %.3f | PeakTime: %.3f\n', ...
        info.RiseTime, info.SettlingTime, info.Overshoot, info.Peak, info.PeakTime);
end

title('Closed Loop: Kd Variation');
legend('0.01','0.05','0.1','0.2','0.5');
saveas(gcf,'results/Kd_variation.png');


%% Open Loop (No Controller)

figure;
step(G);
title('Open Loop Motor Response');
grid on;
saveas(gcf,'results/open_base.png');


%% Final PID Gains

Kp = 3; Ki = 2; Kd = 0.1;
C = pid(Kp,Ki,Kd);


%% Open Loop with PID

OL_final = C * G;

figure;
step(OL_final);
title('Open Loop with PID');
grid on;
saveas(gcf,'results/open_with_pid.png');


%% Closed Loop Final

T_final = feedback(C*G,1);

figure;
step(T_final);
title('Final Closed Loop PID');
grid on;
saveas(gcf,'results/final_closed.png');


%% Comparison Plot

figure;
step(OL_final, T_final);
legend('Open Loop','Closed Loop');
title('Comparison');
grid on;
saveas(gcf,'results/comparison.png');


%% Performance Metrics 

disp("Final Performance:")

info_closed = stepinfo(T_final);

fprintf('\nClosed Loop:\n');
fprintf('Rise: %.3f | Settling: %.3f | Overshoot: %.2f | Peak: %.3f | PeakTime: %.3f\n', ...
    info_closed.RiseTime, info_closed.SettlingTime, info_closed.Overshoot, info_closed.Peak, info_closed.PeakTime);

if isstable(OL_final)
    Yf = dcgain(OL_final);

    if isfinite(Yf)
        info_open = stepinfo(OL_final,'YFinal',Yf);

        fprintf('\nOpen Loop:\n');
        fprintf('Rise: %.3f | Settling: %.3f | Overshoot: %.2f | Peak: %.3f | PeakTime: %.3f\n', ...
            info_open.RiseTime, info_open.SettlingTime, info_open.Overshoot, info_open.Peak, info_open.PeakTime);
    else
        fprintf('\nOpen Loop: No finite steady state\n');
    end
else
    fprintf('\nOpen Loop: Unstable system\n');
end


%% Display Tables 

disp("Kp Results Table:");
disp(array2table(Kp_results, 'VariableNames', ...
{'Kp','RiseTime','SettlingTime','Overshoot','Peak','PeakTime'}));

disp("Ki Results Table:");
disp(array2table(Ki_results, 'VariableNames', ...
{'Ki','RiseTime','SettlingTime','Overshoot','Peak','PeakTime'}));

disp("Kd Results Table:");
disp(array2table(Kd_results, 'VariableNames', ...
{'Kd','RiseTime','SettlingTime','Overshoot','Peak','PeakTime'}));


%% Final Gains

fprintf('\nFinal PID Gains:\nKp = %.2f, Ki = %.2f, Kd = %.2f\n', Kp, Ki, Kd);
figure;
hold on;
grid on;
box on;

[y,t] = step(T_final);
plot(t,y,'b','LineWidth',2);

title('Final Closed Loop PID Response');
xlabel('Time (seconds)');
ylabel('Motor Speed');

info = stepinfo(T_final);

% Rise time marker
rt = info.RiseTime;
y_rt = interp1(t,y,rt);

plot(rt,y_rt,'o','MarkerSize',6,'MarkerFaceColor','b','MarkerEdgeColor','b');
text(rt,y_rt, sprintf('  Rise = %.2fs',rt), ...
    'VerticalAlignment','bottom','FontSize',9);

% Peak marker
[peak_val, idx] = max(y);
peak_time = t(idx);

plot(peak_time,peak_val,'o','MarkerSize',6,'MarkerFaceColor',[0.2 0.2 0.2]);

text(peak_time,peak_val, ...
    sprintf('  Peak = %.2f\n  OS = %.1f%%', peak_val, info.Overshoot), ...
    'VerticalAlignment','bottom','FontSize',9);

saveas(gcf,'results/final_closed_annotated.png');