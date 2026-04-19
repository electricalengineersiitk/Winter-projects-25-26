%% gain_scheduled.m — Gain-Scheduled PID Controller
% EV Throttle Control | End-Term Project | Bhavit Meena (240272)
%
% Purpose:
%   Simulates a gain-scheduled controller that switches PID gains based on
%   the current throttle setpoint zone. Three zones are defined:
%
%   Zone 1 (Low  0–30%):   Gentle creep — lower Kp for smooth low-speed
%   Zone 2 (Mid 30–70%):   Normal drive  — balanced performance
%   Zone 3 (High 70–100%): Acceleration  — higher Kp for fast response
%
% Throttle profile: 20% → 50% → 85%

clear;
clc;
close all;

% % -- -Plant-- - K = 1;
tau = 0.5;
G = tf(K, [tau 1]);

% % -- -Zone PID Gains-- - % Each row : [Kp Ki Kd] zones = struct();
zones(1).name = 'Zone 1: Low (0–30%)';
zones(1).limits = [ 0, 0.30 ];
zones(1).Kp = 1.5;
zones(1).Ki = 0.8;
zones(1).Kd = 0.05;

zones(2).name = 'Zone 2: Mid (30–70%)';
zones(2).limits = [ 0.30, 0.70 ];
zones(2).Kp = 2.5;
zones(2).Ki = 1.0;
zones(2).Kd = 0.10;

zones(3).name = 'Zone 3: High (70–100%)';
zones(3).limits = [ 0.70, 1.00 ];
zones(3).Kp = 4.0;
zones(3).Ki = 1.5;
zones(3).Kd = 0.20;

% % -- -Print Zone Summary-- - fprintf('--- Gain-Scheduled PID Zones ---\n');
for
  i = 1 : 3 fprintf('  %s  |  Kp=%.1f  Ki=%.1f  Kd=%.2f\n', ... zones(i).name,
                    zones(i).Kp, zones(i).Ki, zones(i).Kd);
end

        % % -- -Simulation : Three Throttle Setpoints-- -
    setpoints = [ 0.20, 0.50, 0.85 ];
% normalised throttle targets durations = [ 3.0, 3.0, 3.0 ];
% seconds per zone dt = 0.01;

t_all = [];
y_all = [];
zone_boundaries = [];

t_start = 0;
for
  seg = 1 : length(setpoints) sp = setpoints(seg);

    % Determine which zone this setpoint belongs to
    for z = 1:3
        if sp >= zones(z).limits(1) && sp <= zones(z).limits(2)
            Kp = zones(z).Kp;
    Ki = zones(z).Ki;
    Kd = zones(z).Kd;
    break;
    end end

        C_z = pid(Kp, Ki, Kd);
    T_z = feedback(C_z * G, 1);

    t_seg = (0 : dt : durations(seg) - dt)'; y_seg = sp * step(T_z, t_seg);
    % scale to actual setpoint

            t_all = [t_all; t_seg + t_start];
    % #ok<AGROW> y_all = [y_all; y_seg];
    % #ok<AGROW>

            if seg >
        1 zone_boundaries(end + 1) = t_start;
    % #ok<AGROW> end t_start = t_start + durations(seg);
    end

            % % -- -Build Setpoint Staircase-- -
        sp_all = [];
    t_sp = 0;
for
  seg = 1 : length(setpoints) n = round(durations(seg) / dt);
sp_all = [sp_all; repmat(setpoints(seg), n, 1)];
% #ok<AGROW> end

        % % -- -Plot-- -
    figure('Name', 'Gain-Scheduled Response', 'NumberTitle', 'off');
plot(t_all, y_all, 'g-', 'LineWidth', 2);
hold on;
plot(t_all, sp_all, 'k--', 'LineWidth', 1);
for
  xb = zone_boundaries xline(xb, 'm--', 'Zone Transition',
                             'LabelVerticalAlignment', 'bottom');
end grid on;
xlabel('Time (s)');
ylabel('Motor Speed (normalised)');
title('Gain-Scheduled Throttle Control (3 Zones)');
legend('Motor Speed', 'Setpoint', 'Location', 'southeast');

% % -- -Print Metrics per zone segment-- -
    fprintf('\n--- Per-Segment Performance ---\n');
fprintf('  Segment 1 (Zone 1, sp=%.2f): gains Kp=%.1f Ki=%.1f Kd=%.2f\n',
        ... setpoints(1), zones(1).Kp, zones(1).Ki, zones(1).Kd);
fprintf('  Segment 2 (Zone 2, sp=%.2f): gains Kp=%.1f Ki=%.1f Kd=%.2f\n',
        ... setpoints(2), zones(2).Kp, zones(2).Ki, zones(2).Kd);
fprintf('  Segment 3 (Zone 3, sp=%.2f): gains Kp=%.1f Ki=%.1f Kd=%.2f\n',
        ... setpoints(3), zones(3).Kp, zones(3).Ki, zones(3).Kd);

% % -- -Save-- - saveas(gcf, fullfile(fileparts(mfilename('fullpath')),
                                      'results', 'gainscheduled_response.png'));
fprintf('\nSaved: results/gainscheduled_response.png\n');
