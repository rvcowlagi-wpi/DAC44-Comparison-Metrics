function datagen_linear1()

%% 
close all; clc
rng(3);			% Seed the random number generator

%% Problem setup

tFinal		= 15;
nTimePoints = 1000;
timeStamps	= linspace(0, tFinal, nTimePoints);
tIndexStore	= 1:50:nTimePoints;

wSeriesPC	= [0.1*randn, zeros(1, nTimePoints - 1)];						% piecewise constant

for m1 = 2:nTimePoints
	if mod(m1, 50)
		wSeriesPC(m1) = wSeriesPC(m1 - 1);
	else
		wSeriesPC(m1) = 0.1*randn;
	end
end
wSpline	= spline(timeStamps(25:50:end), wSeriesPC(25:50:end));
wSeries = ppval(wSpline, timeStamps);

%% System state space model
nStates		= 2;
nControl	= 1;

evs = [-1 -2];																% Specified eigenvalues
Q	= orth(randn(nStates));													% Orthonormal basis
D	= diag(evs);															% Construct a diagonal matrix D with the given eigenvalues
A	= Q * D * Q';															% Generate the matrix A using eigenvalue decomposition
B	= randn(nStates, nControl);

alfa1 = 1;
alfa2 = 1;
alfa3 = 1;
alfa4 = 1;


%% System trajectories
tStore	= [];
xStore	= [];
uStore	= [];
for k1 = 9
	rng(k1^2)

	x0			= randn(nStates, 1);
	[~,xSimTrue]= ode45(@system_dynamics_true, timeStamps, x0);
	[~, xSim]	= ode45(@system_dynamics_model, timeStamps, x0);
	
	uSim		= control_input(timeStamps, xSimTrue);
	
	tStore	= [tStore timeStamps(tIndexStore)];
	xStore	= [xStore	(xSimTrue(tIndexStore, :))'];
	uStore	= [uStore	(uSim(tIndexStore, :))'];


	plot(timeStamps, xSimTrue); 
	hold on;
	ax = gca; ax.ColorOrderIndex = ax.ColorOrderIndex - 1;
	plot(timeStamps, xSim, 'LineStyle', '--')
end


%% Functions

	%----- True system dynamics
    function xDot_ = system_dynamics_true(t_, x_)

		u_		= control_input(t_, x_);
        phi_	= [alfa1*sin(alfa3*x_(2)); alfa2*sin(alfa4*x_(1))]; 					% unmodeled dynamics
        xDot_	= A*x_ + B*u_ + phi_ + process_noise(t_);
	end

	%----- Model system dynamics
    function xDot_ = system_dynamics_model(t_, x_)

		u_		= control_input(t_, x_);
        xDot_	= A*x_ + B*u_;
	end

	%----- Process Noise    
    function w_ = process_noise(t_)
        tIndex_ = find(t_ <= timeStamps, 1, 'first');
        w_		= 0; %wSeries(tIndex_);
	end

	%----- Control
	function u_ = control_input(t_, x_)
		u_		= zeros(length(t_), nControl);
	end

end
