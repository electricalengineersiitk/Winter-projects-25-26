clc; clear; close all;

K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

t = 0:0.01:10;

% Define 3 segments
t1 = 0:0.01:3;
t2 = 0:0.01:3;
t3 = 0:0.01:4;

% Zone 1
C1 = pid(2,6,0.1);
T1 = feedback(C1*G,1);
y1 = 0.2 * step(T1, t1);

% Zone 2
C2 = pid(3,5,0.1);
T2 = feedback(C2*G,1);
y2 = 0.5 * step(T2, t2);

% Zone 3
C3 = pid(5,2,0.1);
T3 = feedback(C3*G,1);
y3 = 0.85 * step(T3, t3);

% Combine signals
t_combined = [t1(:); (t2(:)+3); (t3(:)+6)];
y_combined = [y1(:); y2(:); y3(:)];

% Plot
figure;
plot(t_combined, y_combined, 'LineWidth',2);
hold on;
xline(3,'--r','Zone 1→2');
xline(6,'--r','Zone 2→3');

title('Gain Scheduled Response');
xlabel('Time (s)');
ylabel('Speed');

saveas(gcf, 'gainscheduled_response.png');