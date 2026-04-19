%% Conditions we have to keep in mind :
%  •  Rise time < 1 second  (how quickly motor reaches target speed)
%  •  Overshoot < 10%       (how much it exceeds the target before settling)
%  •  Steady-state error < 2%  (how close it gets to the exact target)

K=1 ;                           
tau =0.1 ;                        
G = tf(K,[tau,1]);

%% Applying the auto tuner and verfying the conditions to be satisfied :

C = pidtune(G,'PID')
T = feedback(C*G,1);

% Checking the steady state error 
[y,t] = step(T);
y_ss = y(end);
e_ss = 1-y_ss;

% rise time and overshoot 
info = stepinfo(T);

rise_time = info.RiseTime;
settling_time = info.SettlingTime ;
overshoot = info.Overshoot;

fprintf('Rise Time , Overshoot and Steady State error\n')
fprintf('Rise Time: %.2f seconds\n', rise_time);
fprintf('Overshoot: %.2f%%\n', overshoot);
fprintf('Steady State Error: %.2f%%\n', e_ss * 100);


%% All requirements Full-filled 
% PLOTTING THE STEP RESPONSE OF TUNED TRANSFER FUNCTION

step(T)
title('Step response of Closed loop transfer function')

