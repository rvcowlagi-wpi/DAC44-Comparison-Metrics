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
This program implements the minimum-time solution for the Zermelo
navigation problem. Wind is an external function handle.
%}

function [sim_result_, solution_found_] = mintime_solver(n_traj_points)

close all; clc;

wind_fcn			= @calculate_wind02;
wind_params.const_1 = 2*pi*rand;
wind_params.const_2 = 0.25*rand;

verbose_and_plot_	= false;

%% Problem data

%----- Aircraft speed (normalized units)
spd_V	= 0.05;

%----- Initial and terminal states (position)
x_init	= [0; 0.8];
x_term	= [-0.8; -0.9];
[wind_x1_init, wind_x2_init, ~, ~, ~, ~] = ...
	wind_fcn(x_init(1), x_init(2), wind_params);


%% Find optimal initial heading and traversal time
%{
	There may be multiple solutions, each correspoding to a local minimum.
	We find as many as possible through randomized initial guesses.
%}
ode_solver_options	= odeset('RelTol', 1E-9, 'AbsTol', 1E-9);
bc_solver_options	= optimoptions('fsolve', ...
	'Display', 'none', 'FunctionTolerance', 1E-6);

n_trials	= 50;
min_cost	= Inf;
y_star		= [Inf; Inf];
solution_found_ = 0;
for n_ = 0:n_trials
	clc;

	if verbose_and_plot_
		fprintf('Trial: \t %i\n', n_);
	end

	%----- Initial guesses for optimal initial heading and traversal time tf
	initial_guess_opt_psi0	= 2*pi*rand;
	initial_guess_opt_tf	= (1 + 4*rand)*(norm(x_term - x_init) / spd_V);

	%----- Threat at initial position
	[wind_x1_init, wind_x2_init, ~, ~, ~, ~] = ...
		wind_fcn(x_init(1), x_init(2), wind_params);

	%----- Solve for optimal initial heading and traversal time
	[y_opt, ~, exit_flag]	= ...
		fsolve(@boundary_conditions, ...
		[initial_guess_opt_psi0; initial_guess_opt_tf], bc_solver_options);

	if ~any(exit_flag == [1 2 3 4])
		%---- Boundary conditions not satisfied
		% This is not a valid solution
		continue;
	end
	
	%----- Validate solution: Check if Hamiltonian remains zero always
	psi0_opt= y_opt(1);
	tf_opt	= y_opt(2);
	q_init	= [x_init; psi0_opt; ...
		-cos(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt)); ...
		-sin(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt))];

	[t_sim_traj, q_sim_traj] = ode45(@system_dynamics, ...
		linspace(0, tf_opt, 1E4), q_init, ode_solver_options);

	hamiltonian_traj= zeros(numel(t_sim_traj), 1);
	tr2_sim			= zeros(numel(t_sim_traj), 1);
	for m1 = 1:numel(t_sim_traj)
		x1_t	= q_sim_traj(m1, 1);
		x2_t	= q_sim_traj(m1, 2);
		psi_t	= q_sim_traj(m1, 3);
		p1_t	= q_sim_traj(m1, 4);
		p2_t	= q_sim_traj(m1, 5);

		[wind_x1_t, wind_x2_t, ~, ~, ~, ~] = wind_fcn(x1_t, x2_t, wind_params);

		tr2_sim(m1) = abs(tan(psi_t) - p2_t/p1_t);

		hamiltonian_traj(m1) = 1 + p1_t*(spd_V*cos(psi_t) + wind_x1_t) + ...
			p2_t*(spd_V*sin(psi_t) + wind_x2_t);
	end
	
	%----- Check if Hamiltonian is practically zero
	if max(abs(hamiltonian_traj)) < 1E-7 && tf_opt > 0
		%----- Local extremum found
		cost_traj	= tf_opt;
		if cost_traj < min_cost
			min_cost	= cost_traj;
			y_star		= y_opt;
			solution_found_ = 1;
		end
	end
end

%% Simulate system with optimal control (sanity check)
if solution_found_
	psi0_opt= y_star(1);
	tf_opt	= y_star(2);

	q_init	= [x_init; psi0_opt; ...
		-cos(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt)); ...
		-sin(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt))];
	[t_sim_traj, q_sim_traj] = ...
		ode45(@system_dynamics, linspace(0, tf_opt, n_traj_points), ...
		q_init, ode_solver_options);

	wind_x1_traj		= zeros(n_traj_points, 1); 
	wind_x2_traj		= zeros(n_traj_points, 1);
	wind_grad_11_traj	= zeros(n_traj_points, 1);
	wind_grad_12_traj	= zeros(n_traj_points, 1);
	wind_grad_21_traj	= zeros(n_traj_points, 1); 
	wind_grad_22_traj	= zeros(n_traj_points, 1);

	for m2 = 1:n_traj_points
		[tmp1_, tmp2_, tmp3_, tmp4_, tmp5_, tmp6_] = ...
			wind_fcn(q_sim_traj(m2, 1), q_sim_traj(m2, 2), wind_params);
		wind_x1_traj(m2)		= tmp1_;
		wind_x2_traj(m2)		= tmp2_;
		wind_grad_11_traj(m2)	= tmp3_;
		wind_grad_12_traj(m2)	= tmp4_;
		wind_grad_21_traj(m2)	= tmp5_;
		wind_grad_22_traj(m2)	= tmp6_;
	end
	
	% UNCOMMENT THIS IF WIND GRADIENTS ARE NEEDED
% 	sim_result_	= [x_init; x_term; y_star; min_cost; t_sim_traj; ...
% 		q_sim_traj(:, 1); q_sim_traj(:, 2); ...
% 		q_sim_traj(:, 3); ...
% 		q_sim_traj(:, 4); q_sim_traj(:, 5); ...
% 		wind_x1_traj; wind_x2_traj;
% 		wind_grad_11_traj; wind_grad_12_traj; wind_grad_21_traj; wind_grad_22_traj];

	sim_result_	= [x_init; x_term; y_star; min_cost; t_sim_traj; ...
		q_sim_traj(:, 1); q_sim_traj(:, 2); ...
		q_sim_traj(:, 3); ...
		q_sim_traj(:, 4); q_sim_traj(:, 5); ...
		wind_x1_traj; wind_x2_traj];
else
	sim_result_ = [];
end

if ~verbose_and_plot_
	return
end

%% Plot optimal trajectory
wksp		= 1;
n_plot_pts	= 21;
x1min		= min(q_sim_traj(:, 1)) - 0.1;
x1max		= max(q_sim_traj(:, 1)) + 0.1;
x2min		= min(q_sim_traj(:, 2)) - 0.1;
x2max		= max(q_sim_traj(:, 2)) + 0.1;
[x1, x2]	= meshgrid( ...
	linspace(min(-wksp, x1min), max(wksp, x1max), n_plot_pts), ...
	linspace(min(-wksp, x2min), max(wksp, x2max), n_plot_pts));
[wind_x1_grid, wind_x2_grid, ~, ~, ~, ~] = wind_fcn(x1, x2, wind_params);

figure('units', 'normalized', 'OuterPosition', [0.05 0.05 0.6 0.9]);
hold on; grid on; axis equal; %axis tight;
xlim([min(-wksp, x1min), max(wksp, x1max)]); 
ylim([min(-wksp, x2min), max(wksp, x2max)])
plot(x_init(1), x_init(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
plot(x_term(1), x_term(2), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
text(x_init(1) , x_init(2) + 0.02, 'A', 'FontName', 'Times New Roman', ...
	'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r');
text(x_term(1), x_term(2) + 0.02, 'B', 'FontName', 'Times New Roman', ...
	'FontSize', 30, 'FontWeight', 'bold', 'Color', 'r');

ax = gca;
ax.ColorOrderIndex = 1;
quiver(x1, x2, wind_x1_grid, wind_x2_grid)
plot(q_sim_traj(:, 1), q_sim_traj(:, 2), 'k', 'LineWidth', 6);

ax.FontName = 'Times New Roman';
ax.FontSize = 20;
figtitle	= ['zermelo_' num2str(posixtime(datetime(datestr(now)))) '.png'];
exportgraphics(ax, figtitle, 'Resolution', 300);
xlabel('$p_x$ (normalized units)', 'FontName', 'Times New Roman', ...
	'FontSize', 20, 'FontAngle', 'italic', 'FontWeight', 'bold', 'interpreter', 'latex');
ylabel('$p_y$ (normalized units)', 'FontName', 'Times New Roman', ...
	'FontSize', 20, 'FontAngle', 'italic', 'FontWeight', 'bold', 'interpreter', 'latex'); 


%%	STATE, CONTROL, AND COSTATE DYNAMICS ALONG OPTIMAL TRAJECTORIES
	function q_dot = system_dynamics(t_, q_)
		
		x1_	= q_(1);
		x2_	= q_(2);
		psi_= q_(3);
		p1_	= q_(4);
		p2_	= q_(5);
		
		[wind_x1_, wind_x2_, ...
			wind_gradient_11_, wind_gradient_12_, ...
			wind_gradient_21_, wind_gradient_22_] = ...
			wind_fcn(x1_, x2_, wind_params);
		q_dot(1:2,1)= spd_V*[cos(psi_); sin(psi_)] + [wind_x1_; wind_x2_];	% Aircraft kinematics
		
		q_dot(3, 1)	= wind_gradient_21_*(sin(psi_)^2) + ...
			(wind_gradient_11_ - wind_gradient_22_)*sin(psi_)*cos(psi_) - ...
			wind_gradient_12_*(cos(psi_)^2);								% Optimal control law
		
		q_dot(4:5,1)= -[wind_gradient_11_ wind_gradient_21_; ...
			wind_gradient_12_ wind_gradient_22_]*[p1_; p2_];				% Costate dynamics
	end

%%	STATE AND CONTROL DYNAMICS ALONG OPTIMAL TRAJECTORIES (W/O COSTATE)
	function q_dot = state_dynamics(t_, q_)
		
		x1_	= q_(1);
		x2_	= q_(2);
		psi_= q_(3);
		
		[wind_x1_, wind_x2_, ...
			wind_gradient_11_, wind_gradient_12_, ...
			wind_gradient_21_, wind_gradient_22_] = ...
			wind_fcn(x1_, x2_, wind_params);
		q_dot(1:2,1)= spd_V*[cos(psi_); sin(psi_)] + [wind_x1_; wind_x2_];	% Aircraft kinematics
		
		q_dot(3, 1)	= wind_gradient_21_*(sin(psi_)^2) + ...
			(wind_gradient_11_ - wind_gradient_22_)*sin(psi_)*cos(psi_) - ...
			wind_gradient_12_*(cos(psi_)^2);								% Optimal control law
	end

%%	BOUNDARY CONDITIONS
	function f_	= boundary_conditions(y_)
		
		psi0_	= y_(1);
		tf_		= y_(2);
		
		q_init_	= [x_init; psi0_];
		
		[~, q_sim] = ode45(@state_dynamics, [0 tf_], q_init_, ode_solver_options);
		position_error = q_sim(end, 1:2)' - x_term;
		f_	= position_error;
	end
end

