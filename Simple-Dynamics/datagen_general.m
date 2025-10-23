%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by 
	Raghvendra V. Cowlagi

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
Large trajectory data generator for simple systems with process noise. This
script generates a .mat file. See "datawrite_general.m" for writing .csv
files from this dataset.
%}

function datagen_general()

%% General
close all; clc;

SIM_OPTIONS.dualScreen		= false;

%% Setup
%----- System
id_			= 1;

nState		= [];
tFinal		= [];
fDynamics	= [];
fileprefix	= [];
syspar_		= [];
x0			= [];
system_selector(id_);

%----- Dataset characteristics
nExamples		= 1E3;
nDiscretization	= 100;														% Can be sampled down when writing to csv
dataSize		= nDiscretization*nState;
dt_				= tFinal / nDiscretization;									% Time step

%---- Initialize
trajectoryData = zeros(dataSize, nExamples);
for m = 1:nExamples
	xSim	= zeros(nDiscretization, nState);

	parameter_selector(id_)
	my_simulator()

	trajectoryData(:, m) = xSim(:);
end
tSim	= linspace(0, tFinal, nDiscretization);

%% Calculate distances among examples
maxDistances = zeros(nExamples, 1);
for m1 = 1:nExamples
	tmp1 = zeros(1, nExamples);
	for m2 = (m1 + 1):nExamples
		tmp1(m2) = dt_ * ...	
			sum( (trajectoryData(:, m1) - trajectoryData(:, m2)).^2 );
	end
	maxDistances(m1) = max(tmp1(m1:end));
end

similarityBenchmark = max(maxDistances);
meanTrajectory		= sum(trajectoryData, 2) / nExamples;

histogram(maxDistances)

filename_ = [fileprefix '_all.csv'];
writematrix(trajectoryData', filename_)

return

%% Sanity check
% %---- Take a new sample trajectory
% xSim	= zeros(nDiscretization, nState);
% parameter_selector(id_)
% my_simulator()
% newExample = xSim(:);
% 
% %---- Calculate distance
% dNewToLibrary = zeros(1, nExamples);
% for m1 = 1:nExamples
% 	dNewToLibrary(m1) = dt_ * ...
% 			sum( (trajectoryData(:, m1) - newExample).^2 );
% end
% maxDNew = max(dNewToLibrary);
% 
% isSimilar = (maxDNew <= similarityBenchmark)

%% Plot a few trajectories
nTrajToPlot = randperm(nExamples, 10);


if SIM_OPTIONS.dualScreen
	dispXOffset = 1;
else
	dispXOffset = 0;
end
figure('Name','Multi-Agent Rendezvous', 'Units','normalized', ...
		'OuterPosition', [dispXOffset + 0.05 0.05 0.5 0.3*nState])
ax = gca;
% thisColor	= ax.ColorOrder(1,:);
myFont		= 'Times New Roman';
% 
% grHdlTmp = plot(0,0); hold on; grid on; axis equal
% xlim(1.5*WORKSPACE_SIZE*[-1 1]); ylim(1.5*WORKSPACE_SIZE*[-1 1])
% ax.FontName = myFont;
% ax.FontSize = 16;
% delete(grHdlTmp)
% ax.Units = 'pixels';
% 
% nowText			= num2str(round(posixtime(datetime)));
% videoFileName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.mp4'];
% dataFileName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.mat'];
% firstframeName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.png'];


for m2 = 1:nState
	thisMean	= meanTrajectory((1 + (m2 - 1)*nDiscretization):(m2*nDiscretization));
	subplot(nState, 1, m2); 
	for m1 = nTrajToPlot
		thisState = trajectoryData((1 + (m2 - 1)*nDiscretization):(m2*nDiscretization), m1);
		plot(tSim, thisState); hold on; grid on;
	end
	plot(tSim, thisMean, 'LineWidth', 3)
end

for m2 = 1:nState
	subplot(nState, 1, m2);
	yLabelText	= ['$x_' num2str(m2) '$'];
	ax = gca;
	ax.FontName = myFont;
	ax.FontSize = 16;
	xlabel('$t$', 'Interpreter','latex');
	ylabel(yLabelText, 'Interpreter','latex');
end


% save(filename_,"tSim","trajectoryData")

% prompt_				= ['\n\n ***** Erase all *.csv files from ' foldername_ '? (y/n) ***** \n'];
% confirmDeleteFolder = input(prompt_, "s");

% if strcmp(confirmDeleteFolder, 'y')
% 	delete([foldername_ '/*.csv'])
% else
% 	foldername_ = ['Data/lti_2d_trajectories_' num2str(round(posixtime(datetime)))];
% end

% for k1 = 1:nExamples
% 	traj_k1		= zeros(nDiscretization, nState);
% 	for k2 = 1:nState
% 		traj_k1(:, k2) = trajectoryData( ...
% 			(1 + (k2 - 1)*nDiscretization):(k2*nDiscretization), k1 );
% 	end
% 	
% 	filename_ = [foldername_ '/example' num2str(k1) '.csv'];
% 
% 	writematrix(traj_k1, filename_ );
% end

	%% System selector
	function system_selector(id_)
		switch id_			
			case 1
				nState		= 1;
				tFinal		= 10;
				fDynamics	= @lti1d_uncertainA;
				fileprefix	= 'Data/lti1d_uncertainA';
			case 2
				nState		= 1;
				tFinal		= 10;
				fDynamics	= @lti1d_uncertain_struct;
% 				filename_	= 'Data/lti1d_uncertain_noise';
			case 3
				nState		= 2;
				tFinal		= 10;
				fDynamics	= @lti2d_uncertainA;
% 				filename_	= 'Data/vdp';
% 			case 5
% 				nState		= 2;
% 				tFinal		= 10;
% 				fDynamics	= @my_ipoc;
% 				filename_	= 'Data/ipoc';
% 			case 6
% 				nState		= 2;
% 				tFinal		= 10;
% 				fDynamics	= @lti2d;
% 				filename_	= 'Data/lti2d';
		end
	end

	%% Parameter selector
	function parameter_selector(id_)
		switch id_
			case 1
				x0			= -5 + 10*rand;
				syspar_.a	= -2 + 0.5*randn;
				syspar_.Phi	= exp(syspar_.a*dt_);
			case 2
				x0			= 10*rand; %-5 + 10*rand;
				syspar_.a	= -2 + 0.5*randn;
				syspar_.Phi	= exp(syspar_.a*dt_);
				syspar_.wt	= 1E-2*randn(1, 4);
			case 3
				x0			= 10*rand(2,1); %-5 + 10*rand(2, 1);
				syspar_.A	= [0 1; -5 -1] + [0 0; 1.5*randn 0.3*randn];
				syspar_.Phi	= expm(syspar_.A*dt_);
% 			case 5
% 				syspar_		= [];
% 			case 6
% 				syspar_		= [];
% 			case 7
% 				syspar_		= [];
		end
	end
	
	%% Simulator
	function my_simulator()	
		xk			= x0;
		xSim(1, :)	= xk';
		for k_ = 2:nDiscretization
			xk		= fDynamics(xk);
			xSim(k_, :)	= xk';
		end
	end

	%% LTI 1D with uncertain time constant
	function xk1 = lti1d_uncertainA(xk)
		xk1 = syspar_.Phi*xk;
	end

	%% LTI 1D with uncertain time constant and structured additive uncertainty
	function xk1 = lti1d_uncertain_struct(xk)
		xk1 = syspar_.Phi*xk + syspar_.wt*[sin(xk); cos(xk); sin(2*xk); cos(2*xk)];
	end
	
	%% LTI 2D with uncertain A
	function xk1 = lti2d_uncertainA(xk)
		xk1 = syspar_.Phi*xk;
	end
	
	%% LTI 

end