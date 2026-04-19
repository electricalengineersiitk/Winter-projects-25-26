clear; 
clc;
close all;

K_test = [1, 2, 0.5];
tau = 0.5; 

figure; 
hold on;   % keep all plots

for i = 1:length(K_test)
    
    K = K_test(i);
    
    % Transfer function
    G = tf(K, [tau, 1]);
    
    % Step response plot
    [y, t] = step(G);
    plot(t, y, 'DisplayName', ['K = ' num2str(K,'%.4f')]);
    
    % Step info
    info = stepinfo(G);
    
    rise_time = info.RiseTime;
    settling_time = info.SettlingTime;
    Overshoot = info.Overshoot;
    
    % Steady-state error
    y_ss = y(end);
    e_ss = 1 - y_ss;
    
    % Display results
    fprintf('\n------ For K = %.2f -----\n', K);
    fprintf('Rise Time: %.4f seconds\n', rise_time);
    fprintf('Settling Time: %.4f seconds\n', settling_time);
    fprintf('Overshoot: %.2f %%\n', Overshoot);
    fprintf('Steady State Error: %.4f\n', e_ss);
    
end

grid on;
legend;
title('Step Response for Different K Values');
xlabel('Time (s)');
ylabel('Output');

% changing the values of K gives :
% K = increased to 2 => Steady state error becomes -0.99
% K = decreased to 0.5 => Steady state error became 0.50

% So we can say we are achiving the steady state error zero at the point K=1 


%% Changing tau with fixed value of K :
% changing tau is quite easy as we have to decrease it to make the rise and
% settling time decrease and also it cannot be lower then a particular
% value .


K = 1;                 % fixing the gain
tau_test = [0.5 , 0.1 ,0.01 ,0.001,0.0001 ];

for i = 1:length(tau_test)
    tau = tau_test(i);
    %Transfer function 
    G = tf(K,[tau,1]);
        % Step response plot
    [y, t] = step(G);
    plot(t, y, 'DisplayName', ['tau = ',num2str(tau,'%.4f')]);
    
    % Step info
    info = stepinfo(G);
    
    rise_time = info.RiseTime;
    settling_time = info.SettlingTime;
    Overshoot = info.Overshoot;
    
    % Steady-state error
    y_ss = y(end);
    e_ss = 1 - y_ss;
    
    % Display results
    fprintf('Response for fixed K and varying tau')
    fprintf('\n------ For tau = %.2f -----\n', tau);
    fprintf('Rise Time: %.4f seconds\n', rise_time);
    fprintf('Settling Time: %.4f seconds\n', settling_time);
    fprintf('Overshoot: %.2f %%\n', Overshoot);
    fprintf('Steady State Error: %.4f\n', e_ss);
    
end

%% For now we are choosing the value K= 1 and tau =0.1
%because the responses like given in tau values less then the 0.1 are not
%possible for real systems .

fprintf('\n');
fprintf('The values choosed are K=1 and tau =0.1 for transfer function');
hold off
%Saving the real plot of the transfer functions step response :
K = 1;
tau = 0.1 ;
G = tf(K, [tau,1]);
step(G)
title('Step response of open loop Transfer function')
