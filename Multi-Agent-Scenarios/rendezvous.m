%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by 
	Raghvendra V Cowlagi

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
Multi-agent rendezvous simulation based on 
	M. Ji and M. Egerstedt, "Distributed Coordination Control of Multiagent
	Systems While Preserving Connectedness," in IEEE Transactions on Robotics,
	vol. 23, no. 4, pp. 693-703, Aug. 2007, doi: 10.1109/TRO.2007.900638.

Connectivity-preserving control.
%}

function rendezvous()
close all; clc;

%% Simulation Options and Parameters
WORKSPACE_SIZE		= 1;													% Normalized to 1. Do not change unless there is a really, really good reason
N_AGENTS			= 10;													% Number of agents
MAX_COMMS_DISTANCE	= 0.5;													% Disk size for distance-based comms topology 
DT_					= 1E-3;													% Simulation time step, same as comms. time step
MAX_TRUST_DISTANCE	= MAX_COMMS_DISTANCE*0.9;
DTK_DELAY			= 2;													% Delay (in integer multiples of base time step DT_)

SIM_OPTIONS.measNoise		= false;
SIM_OPTIONS.commsDelay		= true;		
SIM_OPTIONS.commsTopology	= 'distance';									% Options: {'fixed', 'distance'}
SIM_OPTIONS.duration		= 3;
SIM_OPTIONS.dualScreen		= false;
SIM_OPTIONS.makeVideo		= false;
SIM_OPTIONS.saveData		= false;
SIM_OPTIONS.loadInit		= false;

%% Control Parameters
gainK = 3;
controller_ = 'weighted';													% Options: {'plain', 'weighted'}

%% Initialization

if SIM_OPTIONS.loadInit
	load("rendezvous_previous_init.mat", "x_")
else
	%----- Make sure initial agent locations are "spread out" yet connected
	x_		= zeros(2, N_AGENTS);
	x_(:, 1)= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, 1);
	m1 = 2;
	while m1 <= N_AGENTS
		x_(:, m1)	= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, 1);
		dx	= x_(:, m1) - x_(:, 1:m1-1);
		dx	= (dx(1, :).^2 + dx(2, :) .^2).^(0.5);
		if ((m1 <= 4) && any(dx < MAX_TRUST_DISTANCE)) || ...
				((m1 > 4) && any(dx < MAX_TRUST_DISTANCE) && ...
				(sum(dx > 0.9*MAX_COMMS_DISTANCE) > m1 - 3))
			m1 = m1 + 1;
		end
	end

% 	while m1 <= N_AGENTS
% 		x_(:, m1)	= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, 1);
% 		dx	= x_(:, m1) - x_(:, 1:m1-1);
% 		dx	= (dx(1, :).^2 + dx(2, :) .^2).^(0.5);
% 		if ((m1 <= 4) && any(dx < 0.9*MAX_TRUST_DISTANCE)) || ...
% 				((m1 > 4) && any(dx < 0.9*MAX_TRUST_DISTANCE) && ...
% 				(sum(dx > 0.5*MAX_COMMS_DISTANCE) > m1 - 3))
% 			m1 = m1 + 1;
% 		end
% 	end
	save("rendezvous_previous_init.mat", "x_")
end
xCentroid = sum(x_, 2) / N_AGENTS;

t_		= 0;
y_		= x_;
nIter	= 1;
nExpIter= round( SIM_OPTIONS.duration / DT_ );

adjacency_	= zeros(N_AGENTS);
friendship_	= zeros(N_AGENTS);
degree_		= zeros(N_AGENTS);
Laplacian_	= zeros(N_AGENTS);
if strcmp(SIM_OPTIONS.commsTopology, 'distance')
	proximity_graph(0)
end

%----- Store results
tStore	= zeros(nExpIter, 1);
xStore	= zeros(2, N_AGENTS, nExpIter);
amStore	= zeros(N_AGENTS, N_AGENTS, nExpIter);
fmStore	= zeros(N_AGENTS, N_AGENTS, nExpIter);

%% Simulate
while (1)
	tStore(nIter)		= t_;
	xStore(:, :, nIter)	= x_;
	amStore(:, :, nIter)= adjacency_;
	fmStore(:, :, nIter)= friendship_;

	t_	= t_ + DT_;
	%----- Simulate a comms. delay: each agent gets the state from a
	%	neighboring agent from a previous time step (nIter - DTK_DELAY)
	if SIM_OPTIONS.commsDelay && (nIter > DTK_DELAY)
		y_	= xStore(:, :, nIter - DTK_DELAY);
	else
		y_	= x_;
	end

	%----- Simulation termination condition
	if t_ > SIM_OPTIONS.duration + 0.5*DT_
		break;
	end
	
	%----- Check if an equilibrium is reached (consensus or otherwise)
	if nIter > 1
		dx_		= x_ - xStore(:, :, nIter - 1);
		dxNorm	= (dx_(1, :).^2 + dx_(2, :).^2).^(0.5);
		if all(dxNorm < 1E-5), break; end
	end

	nIter	= nIter + 1;

	%----- Update topology if distance-based
	if strcmp(SIM_OPTIONS.commsTopology, 'distance')
		proximity_graph(nIter)
	end

	%----- Update state
	for m1 = 1:N_AGENTS
		xmDot		= agent_dynamics(m1);
		x_(:, m1)	= x_(:, m1) + xmDot*DT_;
	end
end

%% Plot
if SIM_OPTIONS.dualScreen
	dispXOffset = 1;
else
	dispXOffset = 0;
end
figure('Name','Multi-Agent Rendezvous', 'Units','normalized', ...
		'OuterPosition', [dispXOffset + 0.05 0.05 0.5 0.5*16/9])
ax = gca;
thisColor	= ax.ColorOrder(1,:);
myFont		= 'Times New Roman';

grHdlTmp = plot(0,0); hold on; grid on; axis equal
xlim(1.5*WORKSPACE_SIZE*[-1 1]); ylim(1.5*WORKSPACE_SIZE*[-1 1])
ax.FontName = myFont;
ax.FontSize = 16;
delete(grHdlTmp)
ax.Units = 'pixels';

nowText			= num2str(round(posixtime(datetime)));
videoFileName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.mp4'];
dataFileName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.mat'];
firstframeName	= ['Results/202305/rendezvous_' controller_ '_run' nowText '.png'];


if SIM_OPTIONS.makeVideo
	vid_	= VideoWriter(videoFileName, "MPEG-4");
	vid_.open();
end
nSkip	= max(1, round(nIter / 500));
for n1 = 1:nSkip:nIter
	cla();
	plot_network(n1);
	plot_nodes(n1);

	text(-1.42*WORKSPACE_SIZE, 1.42*WORKSPACE_SIZE, ['$t = $' num2str(tStore(n1))], ...
		"FontName", myFont, 'FontSize', 16, 'Interpreter', 'latex')
	drawnow();

	if SIM_OPTIONS.makeVideo
		ax = gca;
% 		vid_.writeVideo(getframe(ax, [-50 -50 ax.Position(3) + 80 ax.Position(4) + 50]));
		vid_.writeVideo(getframe(ax, [-50 -50 ax.Position(3) + 80 ax.Position(4)]));
		if n1 == 1
			exportgraphics(gcf,firstframeName, 'Resolution', 300)
		end
	end
end
if SIM_OPTIONS.makeVideo, vid_.close(); end

if SIM_OPTIONS.saveData, save(dataFileName); end

	%% Distance-based comms	network graph
	function proximity_graph(ell_)
		adjacency_	= zeros(N_AGENTS);
		friendship_	= zeros(N_AGENTS);
		for m2 = 1:N_AGENTS
			distance_ = x_ - x_(:, m2);
			distance_ = ( distance_(1, :).^2 + distance_(2, :).^2 ).^(0.5);
			isProximal= distance_ <= MAX_COMMS_DISTANCE;

			adjacency_(m2, isProximal)	= 1;
			adjacency_(m2, m2)			= 0;

			if ell_ > 1
				isFriend= (distance_ <= MAX_TRUST_DISTANCE) | ...
					(isProximal & fmStore(m2, :, ell_ - 1)); 
			else
				isFriend= distance_ <= MAX_TRUST_DISTANCE;
			end			
			friendship_(m2, isFriend)	= 1;
			friendship_(m2, m2)			= 0;
		end
		degree_		= diag(sum(adjacency_, 1));
		Laplacian_	= degree_ - adjacency_;		
	end

	%% Draw network topology
	function plot_network(ell_)
		for m2 = 1:N_AGENTS
			for m3 = (m2 + 1) : N_AGENTS
				if ~amStore(m2, m3, ell_), continue; end
				line([xStore(1, m2, ell_), xStore(1, m3, ell_)], ...
					[xStore(2, m2, ell_), xStore(2, m3, ell_)]);
			end
		end
	end

	%% Draw nodes
	function plot_nodes(ell_)
		plot(xStore(1, :, ell_), xStore(2, :, ell_), ...
			'Marker','o', 'MarkerFaceColor', thisColor, ...
			'MarkerEdgeColor', thisColor, 'MarkerSize', 25, ...
			'LineStyle','none');

		for m2 = 1:N_AGENTS
			textXOffset = 0.01;
			if length(num2str(m2)) > 1, textXOffset = textXOffset + 0.02; end
			text(xStore(1, m2, ell_) - textXOffset, xStore(2, m2, ell_), ...
				num2str(m2), ...
				'FontSize', 13, 'FontName', 'Times New Roman', ...
				'FontWeight', 'bold', 'Color', 'w')
		end

		plot(xCentroid(1), xCentroid(2), ...
			'Marker','diamond', 'MarkerFaceColor', ax.ColorOrder(2,:), ...
			'MarkerEdgeColor', ax.ColorOrder(2,:), 'MarkerSize', 18, ...
			'LineStyle','none');
	end

	%% Agent dynamics
	function xmDot = agent_dynamics(m_)
		%-- Find friends
		tmp_		= 1:N_AGENTS;
		friendAgents= tmp_(friendship_(m_, :) > 0);

		%-- Update based on friend data
		um_ = 0;
		if strcmp(controller_, 'plain')
			for m2 = friendAgents
				um_ = um_ - gainK*(y_(:, m_) - y_(:, m2));
			end
		else
			for m2 = friendAgents
				distanceToFriend = norm( y_(:, m_) - y_(:, m2) );
				if abs(MAX_COMMS_DISTANCE - distanceToFriend) > 0.1*MAX_COMMS_DISTANCE
					um_ = um_ - ...
						(2*MAX_COMMS_DISTANCE - 2*distanceToFriend) * ...
						(y_(:, m_) - y_(:, m2)) / ...
						((MAX_COMMS_DISTANCE - distanceToFriend)^2);
				else
					um_ = um_ - gainK*(y_(:, m_) - y_(:, m2));
				end
			end
		end
		xmDot = um_;
	end

end
