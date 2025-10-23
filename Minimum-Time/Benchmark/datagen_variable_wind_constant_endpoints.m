%{
SOFTWARE LICENSE
----------------
Copyright (c) 2021 by Raghvendra V. Cowlagi

Permission is hereby granted to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in
the Software, including the rights to use, copy, modify, merge, copies of
the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:  

* The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
* The Software, and its copies or modifications, may not be distributed,
published, or sold for profit. 
* The Software, and any substantial portion thereof, may not be copied or
modified for commercial or for-profit use.

The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other
dealings in the software.      


PROGRAM DESCRIPTION
-------------------
This script repeats the minimum time base simulator. The endpoints are
constant while the wind is varied at each trial.

NOTE: This script will take a long time to run, typically around 2s per
trajectory.
%}

clear variables; close all; clc;

%------ For traditional Zermelo solutions
nTrajectories	= 1%10000;
nDiscretization	= 100;
% size_data	= nDiscretization*12 + 7; 
size_data	= nDiscretization*8 + 7; % <==== REMOVING WIND GRADIENTS FROM DATA
%{
	Syntax for each sim_result_ column: 
		[x_init; x_term; y_star; min_cost;	(7 numbers)
		t_sim;								(trajectory time stamps)
		x1_sim; x2_sim; psi_sim;			(position coordinates and heading)
		p1_sim; p2_sim;						(costates)
		wind_x1; wind_y1;					(wind velocity coordinates)
		wind_grad_11; wind_grad_12; wind_grad_21; wind_grad_22] <=====

 Wind velocity and gradient are given at each point along the
 trajectory.
%}
baseline_data = zeros(size_data, nTrajectories);
remove_trials = false(1,nTrajectories);
for m = 1:nTrajectories
	disp(m)
	[sim_result_, solution_found_] = mintime_solver(nDiscretization);
	if solution_found_
		baseline_data(:, m) = sim_result_;
	else
		remove_trials(m)	= true;
	end
end
baseline_data(:, remove_trials) = [];	% remove trials where no solution was found



% n_traj_examples can differ slightly from n_trials if solutions are not
% found for some trials

% foldername_ = ['../Min-Time-Gan/trajectory-examples/n' ...
% 	num2str(n_trials, '%.4i') '_t' num2str(posixtime(datetime(datestr(now))))];
% mkdir(foldername_)

% save dataset3_var_wind.mat baseline_data

% Write data to CSV files using datagen_training.m

return

%% Plot a randomly chosen trajectory
% this_trial	= 1 + round(rand*(nDiscretization - 1));
% filename_	= [foldername_ '/mintime_points' num2str(this_trial) '.csv'];
% this_traj	= readmatrix(filename_);

this_traj	= baseline_data(8+nDiscretization:end, 1);
this_traj	= reshape(this_traj, nDiscretization, 7);

wksp		= 1;
n_plot_pts	= 21;
x1min		= min(this_traj(:, 1)) - 0.1;
x1max		= max(this_traj(:, 1)) + 0.1;
x2min		= min(this_traj(:, 2)) - 0.1;
x2max		= max(this_traj(:, 2)) + 0.1;

figure('units', 'normalized', 'OuterPosition', [0.05 0.05 0.6 0.9]);
hold on; grid on; axis equal; %axis tight;
xlim([min(-wksp, x1min), max(wksp, x1max)]); 
ylim([min(-wksp, x2min), max(wksp, x2max)]);


x_init = this_traj(1, 1:2);
x_term = this_traj(end, 1:2);

plot(x_init(1), x_init(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
plot(x_term(1), x_term(2), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
text(x_init(1) , x_init(2) + 0.02, 'A', 'FontName', 'Times New Roman', ...
	'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r');
text(x_term(1), x_term(2) + 0.02, 'B', 'FontName', 'Times New Roman', ...
	'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r');

ax = gca;
ax.ColorOrderIndex = 1;
quiver(this_traj(1:10:end, 1), this_traj(1:10:end, 2), ...
	this_traj(1:10:end, 6), this_traj(1:10:end, 7))
plot(this_traj(:, 1), this_traj(:, 2), 'k', 'LineWidth', 6);

ax.FontName = 'Times New Roman';
ax.FontSize = 20;
figtitle	= ['zermelo_' num2str(posixtime(datetime(datestr(now)))) '.png'];
% exportgraphics(ax, figtitle, 'Resolution', 300);
xlabel('$p_x$ (normalized units)', 'FontName', 'Times New Roman', ...
	'FontSize', 20, 'FontAngle', 'italic', 'FontWeight', 'bold', 'interpreter', 'latex');
ylabel('$p_y$ (normalized units)', 'FontName', 'Times New Roman', ...
	'FontSize', 20, 'FontAngle', 'italic', 'FontWeight', 'bold', 'interpreter', 'latex'); 

return

%% Zero Hamiltonian and psi-dot tests for trajectory

psi_				= this_traj(:, 3);
p1_					= this_traj(:, 4);
p2_					= this_traj(:, 5);
wind_x1_			= this_traj(:, 6);
wind_x2_			= this_traj(:, 7);
wind_gradient_11_	= this_traj(:, 8);
wind_gradient_12_	= this_traj(:, 9);
wind_gradient_21_	= this_traj(:, 10);
wind_gradient_22_	= this_traj(:, 11);
time_				= this_traj(:, 12);

psi_dot_diff		= ( psi_(2:end) - psi_(1:end-1) ) ./ ...
	( time_(2:end) - time_(1:end-1) );
psi_dot_law			= wind_gradient_21_.*(sin(psi_).^2) + ...
	(wind_gradient_11_ - wind_gradient_22_).*sin(psi_).*cos(psi_) - ...
	wind_gradient_12_.*(cos(psi_).^2);

psi_dot_error		= abs(psi_dot_diff - psi_dot_law(2:end));
max(psi_dot_error)

spd_V	= 0.05; % this is constant
hamiltonian_along_traj	= 1 + p1_.*(spd_V*cos(psi_) + wind_x1_) + ...
			p2_.*(spd_V*sin(psi_) + wind_x2_);
max(abs(hamiltonian_along_traj))

figure('units', 'normalized', 'OuterPosition', [0.05 0.45 0.25 0.5]);
subplot(211); plot(time_(2:end), psi_dot_error, 'LineWidth', 3); ylim([-1E-2 1E-2])
xlabel('Time (s)'); ylabel('$|\dot{\psi}|$ error', 'Interpreter','latex');
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 12;

subplot(212); plot(time_, hamiltonian_along_traj, 'LineWidth', 3); ylim([-1E-6 1E-6])
xlabel('Time (s)'); ylabel('Hamiltonian $|H|$', 'Interpreter','latex')
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 12;