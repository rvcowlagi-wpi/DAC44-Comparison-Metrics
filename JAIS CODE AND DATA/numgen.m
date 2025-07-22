% This function finds the numerical solution(numerically simulated trajectory) given initial conditions 
function [q_sim_traj,X1,Y1,X,y_opt] = numgen(traj)
global x_init;
global x_term;
global wind_params;
global wind_fcn;
global spd_V;
R = traj;
X1 = R(1:25); 
Y1 = R(26:50);
psi = R(51:75);
p1 = R(76:100);
p2 = R(101:125);
W1 = R(126:150); 
W2 = R(151:175);
dt = R(176:200);
G = R(201:end);
X_wind = G(1:25); Y_wind = G(26:50); Wx = G(51:75); Wy = G(76:100); Wx11 = G(101:125); Wx12 = G(126:150);
Wy11 = G(151:175); Wy12 = G(176:200);
X = [X_wind, Y_wind, Wx, Wy, Wx11, Wx12, Wy11, Wy12];

% Define the function
f = @(x, params) [params(2)*(-sin(params(1))*x(:,1) + cos(params(1))*x(:,2))*cos(params(1)), ...
                  params(2)*(-sin(params(1))*x(:,1) + cos(params(1))*x(:,2))*sin(params(1)), ...
                  -params(2) * cos(params(1)) * sin(params(1)) * ones(size(x(:,1))), ...
                  params(2) * cos(params(1)) * cos(params(1)) * ones(size(x(:,1))), ...
                  -params(2) * sin(params(1)) * sin(params(1)) * ones(size(x(:,1))), ...
                  params(2) * cos(params(1)) * sin(params(1)) * ones(size(x(:,1)))];

% Example data
xData = [X(:,1), X(:,2)];  % input data
yData = [X(:,3), X(:,4), X(:,5), X(:,6), X(:,7), X(:,8)];  % corresponding output data

% Provide a range of initial guesses
initial_guesses = [0.1, 0.1;  % Example initial guess 1
                   0.5, 0.5;  % Example initial guess 2
                   1.0, 1.0;
                   2.0,2.0;
                   4.0,4.0;
                   6.0,6.0]; % Example initial guess 3

best_params = [];
best_residuals = Inf;

% Iterate over different initial guesses
for i = 1:size(initial_guesses, 1)
    initialGuess = initial_guesses(i, :);
    % Define options for lsqcurvefit
    options = optimset('Display', 'iter');
    % Use lsqcurvefit to fit the model to the data
    [estimatedParams, residuals, ~, exitflag, output] = lsqcurvefit(@(params, x) f(x, params), initialGuess, xData, yData, [], [], options);
    
    % Check if the current fit is better than the previous best
    if norm(residuals) < best_residuals
        best_params = estimatedParams;
        best_residuals = norm(residuals);
    end
end

% Display results for the best fit
disp('Best Estimated Parameters:');
disp(best_params);
disp('Best Residuals:');
disp(best_residuals);
disp('Exit Flag:');
disp(exitflag);
disp('Output Information:');
disp(output);

% Numerical Trajectory generation 
global wind_fcn;
wind_fcn			= @calculate_wind02;
global wind_params;
wind_params.const_1 = best_params(1);
wind_params.const_2 = best_params(2);
verbose_and_plot_	= false;
%----- Aircraft speed (normalized units)
global spd_V;
spd_V	= 0.05;
n_traj_points = 100;
%----- Initial and terminal states (position)
global x_init;global x_term;
%x_init	= [0; 0.8];
x_init = [X1(1);Y1(1)];
%x_term	= [-0.8; -0.9];
x_term	= [X1(end); Y1(end)];
[wind_x1_init, wind_x2_init, ~, ~, ~, ~] = ...
	wind_fcn(x_init(1), x_init(2), wind_params);

bc_solver_options	= optimoptions('fsolve', ...
	'Display', 'none', 'FunctionTolerance', 1E-6);

n_trials	= 50;
min_cost	= Inf;
y_star		= [Inf; Inf];
solution_found_ = 0;
% Plot the trajectory
% [wind_x1_, wind_x2_, wx11, wx12, wy11, wy12] = ...
% 		wind_fcn(X_wind, Y_wind, wind_params);
 


for n_ = 0:n_trials
	clc;

	if verbose_and_plot_
		fprintf('Trial: \t %i\n', n_);
	end

	%----- Initial guesses for optimal initial heading and traversal time tf
	%initial_guess_opt_psi0	= 2*pi*rand;
    initial_guess_opt_psi0	= psi(1);
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
    ode_solver_options	= odeset('RelTol', 1E-9, 'AbsTol', 1E-9);
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
	disp('Reached point C');
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
	disp('Reached point E');
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
	ode_solver_options	= odeset('RelTol', 1E-9, 'AbsTol', 1E-9);

	psi0_	= y_(1);
	tf_		= y_(2);
	
	q_init_	= [x_init; psi0_];
	
	[~, q_sim] = ode45(@state_dynamics, [0 tf_], q_init_, ode_solver_options);
	position_error = q_sim(end, 1:2)' - x_term;
	f_	= position_error;
end


end