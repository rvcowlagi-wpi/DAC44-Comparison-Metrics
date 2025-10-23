clear variables; close all; clc
addpath(genpath('My-Classes'))


% nTrials	= 1000;
% cd Data
% foldername_ = string( datetime('now', 'Format', 'yyyyMMdd-HHmm') );
% 
% mkdir(foldername_)
% cd(foldername_ )

% parpool('local',16)

% parfor m0 = 1:nTrials
for m0 = 1
	clc;
	fprintf('\n-- Trial: %i \n', m0)

	%----- Problem dimensions
	N_THREAT_STATE	= 10 + round(10*rand);
	N_GRID_ROW		= 25;
	SENSOR_NOISE_VAR= 0.1;	% Variance of (i.i.d.) measurement noise in each sensor, assuming homogeneous sensors
	
	
	grid_		= ACEGridWorld(1, N_GRID_ROW);
	threat_		= ParametricThreat(N_THREAT_STATE, ...
		grid_.halfWorkspaceSize, SENSOR_NOISE_VAR, grid_);
	grid_.threatModel	= threat_;
	
	flags_.SHOW_TRUE	= true;
	flags_.SHOW_ESTIMATE= false;
	flags_.JUXTAPOSE	= true;
	flags_.SHOW_PATH	= false;
	flags_.DUAL_SCREEN	= true;
	
	

	locations		= zeros(2, grid_.nPoints);
	threatValues	= zeros(1, grid_.nPoints);
	minCostToGoal	= zeros(1, grid_.nPoints);
	threatGradient	= zeros(2, grid_.nPoints);
	dx_				= grid_.halfWorkspaceSize*0.01;

	for m1 = 1:1%grid_.nPoints
		thisLocation		= grid_.coordinates(:, m1);
		thisThreatValue		= grid_.threatModel.calculate_at_locations(...
			thisLocation, grid_.threatModel.stateHistory(:, 1) );

		%---- Find minimum threat path
		grid_.searchSetup.start		= m1;
% 		grid_.searchSetup.locationGoal= m1;
		grid_				= grid_.min_cost_path();
	
		locations(:, m1)	= thisLocation;
		threatValues(m1)	= thisThreatValue;

		minCostToGoal(m1)	= grid_.pathCost;

		xPlusdx1	= thisLocation + [dx_; 0];
		xPlusdx2	= thisLocation + [0; dx_];
	
	
		cPlusdx1	= grid_.threatModel.calculate_at_locations(...
			xPlusdx1, grid_.threatModel.stateHistory(:, 1) );
		cPlusdx2	= grid_.threatModel.calculate_at_locations(...
			xPlusdx2, grid_.threatModel.stateHistory(:, 1) );
	
		threatGradient(:, m1) = ([cPlusdx1; cPlusdx2] - thisThreatValue)/dx_;

		grid_.plot_parametric(threat_, [], flags_)
	end
		
	threatDataTable = table(...
		locations(1, :)', locations(2, :)', threatValues', ...
		threatGradient(1, :)', threatGradient(2, :)', minCostToGoal', ...
		'VariableNames',{'x_1 Coordinate', 'x_2 Coordinate', 'Threat Intensity', ...
		'Threat Gradient x_1', 'Threat Gradient x_2', 'Value'});

% 	xPlot	= grid_.coordinates(1, 1:grid_.nGridRow);
% 	yPlot	= grid_.coordinates(2, 1:grid_.nGridRow:grid_.nPoints);
% 	valueMesh	= reshape(minCostToGoal', grid_.nGridRow, grid_.nGridRow);
% 	grHdlSurf	= surfc(xPlot, yPlot, valueMesh,'LineStyle','none');
% 	colorbar; view(2);
% 	axis equal; axis tight; hold on;

% 	filename_	= [num2str(m0,'%03.f'), '.csv'];	
% 	writetable(threatDataTable, filename_, 'Delimiter', ',');
	
end
% cd ../..