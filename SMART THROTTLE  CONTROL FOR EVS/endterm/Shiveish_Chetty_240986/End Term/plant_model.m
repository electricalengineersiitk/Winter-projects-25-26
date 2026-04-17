%Define Parameters
K=1;
tau=0.5;

%Transfer Function
s = tf('s');
G = K / (tau*s + 1);

%Plot Step-Response
figure;
step(G);
title('Open-Loop Response');
xlabel('Time(seconds)');
ylabel('Motor Speed');
% Save
saveas(gcf, 'open_loop_response.png');