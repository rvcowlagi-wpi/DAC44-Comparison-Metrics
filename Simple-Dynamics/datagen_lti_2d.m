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
This script generates trajectory examples of a one-state stable LTI system.
The time constant and initial condition are randomly chosen. The duration
of the trajectories is fixed at 10s.
%}

clear variables; close all; clc;

n_trials	= 70;
n_traj_pts	= 100;
n_state		= 2;
size_data	= n_traj_pts*n_state;
baseline_data = zeros(size_data, n_trials);
remove_trials = false(1,n_trials);
for m = 1:n_trials
	x0	= -5 + 10*rand(n_state, 1);
	A	= [0 1; -5 -1];

	[t_sim, x_sim] = ode45(@(t,x) A*x, linspace(0,10,n_traj_pts), x0);
	baseline_data(:, m) = x_sim(:);
end


n_traj_examples = n_trials; 
% n_traj_examples can differ slightly from n_trials if solutions are not
% found for some trials

% foldername_ = ['Min-Time-GAN/RVC-Sandbox/LTI-1D-trajectory-examples/n' ...
% 	num2str(n_trials, '%.4i') '_t' num2str(posixtime(datetime(datestr(now))))];
% mkdir(foldername_)

foldername_ = 'Data/lti_2d_trajectories';
delete([foldername_ '/*.csv'])

for k1 = 1:n_traj_examples
	traj_k1		= baseline_data(:, k1 );	
	filename_ = [foldername_ '/lti_2d_points' num2str(k1) '.csv'];

	writematrix(traj_k1, filename_ );
end

return

%% Plot a randomly chosen trajectory
this_trial	= 1 + round(rand*(n_traj_examples - 1));

% filename_	= [foldername_ '/lti_1d_points' num2str(this_trial) '.csv'];
% this_traj	= readmatrix(filename_);

this_traj = traj_k1;


c = polyfit(t_sim, this_traj(:, 1), 10);
p = polyval(c, t_sim);

figure('units', 'normalized', 'OuterPosition', [0.05 0.05 0.6 0.6]);
hold on; grid on; axis equal; %axis tight;
% plot(t_sim, this_traj(:, 1), t_sim, p, 'LineWidth', 2)
plot(t_sim, this_traj(:, 1), 'LineWidth', 2); hold on;
plot(t_sim(1:5:end), this_traj(1:5:end, 1), '.', 'MarkerSize', 40)
xlabel('Time~~$t$','Interpreter','latex');
ylabel('Output~~$y_2$', 'Interpreter', 'latex'); xlim([0 10])
ax = gca;
ax.FontName = 'Arial';
ax.FontSize = 20;
return

this_trial = unique( 1 + round(rand(50, 1)*(n_traj_examples - 1)) );
for m1 = 1:3
	for m2 = 1:6
		indx = 6*(m1 - 1) + m2;
		filename_	= [foldername_ '/lti_1d_points' num2str(this_trial(indx)) '.csv'];
		this_traj	= readmatrix(filename_);

		subplot(3, 6, indx)
		plot(t_sim, this_traj(:, 1), 'LineWidth', 2)
		xlabel('$t$', 'Interpreter','latex')
		ylabel('$x_1$', 'Interpreter','latex')
	end
end
exportgraphics(gcf, 'lti_2d_real_examples.png')
