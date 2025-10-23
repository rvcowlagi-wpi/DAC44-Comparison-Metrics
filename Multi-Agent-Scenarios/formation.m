%% Formation Control Problem

function formation()
clc; close all; 

%% Parameters
WORKSPACE_SIZE          = 2;
N_AGENTS                = 9;                                                    
MAX_COMMS_DISTANCE      = 4;   % tri = 0.5 / sq = 0.6 / pen = 0.7 / hex = 0.7	                                                
DT_                     = 1E-3;                                                
MAX_TRUST_DISTANCE      = MAX_COMMS_DISTANCE;
DTK_DELAY               = 0;												


%% Simulation Options
SIM_OPTIONS.measNoise           = false;
SIM_OPTIONS.commsDelay          = true;		
SIM_OPTIONS.commsTopology       = 'distance';								
SIM_OPTIONS.duration            = 5;
SIM_OPTIONS.dualScreen          = false;
SIM_OPTIONS.makeVideo           = false; % turn on/off2 .mp4
SIM_OPTIONS.saveData            = false;
SIM_OPTIONS.loadInit            = false;


%% Initialization

x_ = zeros(2, N_AGENTS);

% % Define the initial positions for each agent			% <<<====== 
%agentPositions = [-3, 2.7; -2.6, 2.9; -2.8, 3.1];                           % triange 
%agentPositions = [-3, 2.6; -2.5, 2.8; -3.1, 3; -2.6, 3];                    % square
%agentPositions = [-3.1, 2.5; -2.6, 2.5; -3.3, 2.9; -2.4, 2.8; -2.8, 3.25];   % pentagon
% agentPositions = [-3.1, 2.6; -2.3, 2.8; -3.2, 3; -2.35, 3.15;...            % hexagon
%                   -2.8, 3.3; -2.8, 2.4]; 

% UPDATED INITIALIZATION TO MAKE ALL AGENTS START AT ORIGIN (ABOVE) OR
% ALONG A STRAIGHT LINE (BELOW)

for m1 = 2:N_AGENTS
	x_(1, m1) = -0.5 + ((m1 - 1) / (N_AGENTS - 1));
end


%----- Desired absolute positions of agents
angle_	= 2 * pi / N_AGENTS;
radius_ = 0.5;
for m1 = 1:N_AGENTS
	desiredAbsolutePosn(:, m1) = ...
		[radius_ * cos((m1 - 1) * angle_); radius_ * sin((m1 - 1) * angle_)];
end
                   

% % Assign the specified positions to the agents
% for m1 = 1:N_AGENTS
%     x_(:, m1) = agentPositions(m1, :)';
% end

t_			= 0;
y_			= x_;
nIter		= 1;
nExpIter	= round(SIM_OPTIONS.duration / DT_);

adjacency_	= zeros(N_AGENTS);
friendship_	= zeros(N_AGENTS);

if strcmp(SIM_OPTIONS.commsTopology, 'distance')
    proximity_graph(0)
end

% Store Results
tStore      = zeros(nExpIter, 1);
xStore      = zeros(2, N_AGENTS, nExpIter);
amStore     = zeros(N_AGENTS, N_AGENTS, nExpIter);
fmStore     = zeros(N_AGENTS, N_AGENTS, nExpIter);


%% Simulate
while (1)
    tStore(nIter)        = t_;
    xStore(:, :, nIter)  = x_;
    amStore(:, :, nIter) = adjacency_;
    fmStore(:, :, nIter) = friendship_;

    t_  = t_ + DT_;
    if SIM_OPTIONS.commsDelay && (nIter > DTK_DELAY)
        y_  = xStore(:, :, nIter - DTK_DELAY);
    else
        y_  = x_;
    end
    if t_ > SIM_OPTIONS.duration + 0.5*DT_
        break;
    end

    % Check if an equilibrium is reached (consensus or otherwise)
    if nIter > 1
        dx_     = x_ - xStore(:, :, nIter - 1);
        dxNorm  = (dx_(1, :).^2 + dx_(2, :).^2).^(0.5);
        if all(dxNorm < 1E-5), break; end
    end

    nIter   = nIter + 1;

    % Update topology if distance-based
    if strcmp(SIM_OPTIONS.commsTopology, 'distance')
        proximity_graph(nIter)
    end

    % Update state
    for m1 = 1:N_AGENTS
        xmDot       = formation_control(m1);
        x_(:, m1)   = x_(:, m1) + xmDot*DT_;
    end
end

%% Plot
% ======== DO NOT CHANGE THIS PLOTTING CODE ===========
if SIM_OPTIONS.dualScreen
    dispXOffset = 1;
else
    dispXOffset = 0;
end
figure('Name', 'Multi-Agent Formation', 'Units', 'normalized', ...
    'OuterPosition', [dispXOffset + 0.05 0.05 0.5 0.5*16/9])
ax = gca;
thisColor   = ax.ColorOrder(1,:);
myFont      = 'Times New Roman';

grHdlTmp = plot(0, 0); hold on; grid on; axis equal
xlim(WORKSPACE_SIZE*[-1 1]); ylim(WORKSPACE_SIZE*[-1 1])
ax.FontName = myFont;
ax.FontSize = 16;
delete(grHdlTmp)
ax.Units = 'pixels';

nowText         = num2str(round(posixtime(datetime)));
videoFileName   = ['formation_' '_run' nowText '.mp4'];

if SIM_OPTIONS.makeVideo
    vid_ = VideoWriter(videoFileName, "MPEG-4");
    vid_.open();
end
nSkip = max(1, round(nIter / 500));
for n1 = 1:nSkip:nIter
    cla();
    plot_network(n1);
    plot_nodes(n1);

    text(-0.92*WORKSPACE_SIZE, 0.92*WORKSPACE_SIZE, ['$t = $' num2str(tStore(n1))], ...
        'FontName', myFont, 'FontSize', 16, 'Interpreter', 'latex')
    drawnow();

    if SIM_OPTIONS.makeVideo
        ax = gca;
        vid_.writeVideo(getframe(ax, [-50 -50 ax.Position(3) + 80 ax.Position(4) + 50]));
    end

end

if SIM_OPTIONS.makeVideo
    vid_.close();
end

if SIM_OPTIONS.saveData
    save('formation_control_data.mat');
end

	%% Distance-based comms	network graph
	function proximity_graph(ell_)
    	adjacency_  = zeros(N_AGENTS);
    	friendship_ = zeros(N_AGENTS);
    	for m2 = 1:N_AGENTS
        	distance_ = x_ - x_(:, m2);
        	distance_ = (distance_(1, :).^2 + distance_(2, :).^2).^(0.5);
        	isProximal = distance_ <= MAX_COMMS_DISTANCE;
	
        	adjacency_(m2, isProximal)	= 1;
			adjacency_(m2, m2)			= 0;% <<<====== 
	
        	if ell_ > 1
            	isFriend = (distance_ <= MAX_TRUST_DISTANCE) | ...
                	(isProximal & fmStore(m2, :, ell_ - 1)); 
        	else
            	isFriend = distance_ <= MAX_TRUST_DISTANCE;
        	end
	
        	friendship_(m2, isFriend)	= 1;
			friendship_(m2, m2)			= 0; % <<<====== 
    	end
	
    	% Set diagonal elements to zero		% <<<====== 
%     	adjacency_(1:N_AGENTS+1:end) = 0;	% <<<====== 
%     	friendship_(1:N_AGENTS+1:end) = 0;	% <<<====== 
		% THIS WON'T SET THE DIAGONALS TO ZERO. FIXED BY SETTING ZEROS
		% ELEMENTWISE ABOVE.
	end

	%% Draw network topology
	function plot_network(ell_)
    	for m2 = 1:N_AGENTS
        	for m3 = (m2 + 1) : N_AGENTS
            	if ~amStore(m2, m3, ell_)
                	continue;
            	end
            	line([xStore(1, m2, ell_), xStore(1, m3, ell_)], ...
                	[xStore(2, m2, ell_), xStore(2, m3, ell_)]);
        	end
    	end
	end

	%% Draw nodes
    function plot_nodes(ell_)
        plot(xStore(1, :, ell_), xStore(2, :, ell_), ...
            'Marker', 'o', 'MarkerFaceColor', thisColor, ...
            'MarkerEdgeColor', thisColor, 'MarkerSize', 25, ...
            'LineStyle', 'none');

        for m2 = 1:N_AGENTS
            textXOffset = 0.01;
            if length(num2str(m2)) > 1
                textXOffset = textXOffset + 0.02;
            end
            text(xStore(1, m2, ell_) - textXOffset, xStore(2, m2, ell_), ...
                num2str(m2), ...
                'FontSize', 13, 'FontName', 'Times New Roman', ...
                'FontWeight', 'bold', 'Color', 'w')
		end

		plot(desiredAbsolutePosn(1, :), desiredAbsolutePosn(2, :), ...
        	'Marker', 'o', 'MarkerSize', 20, 'LineStyle', 'none');
	end

	%% Agent dynamics
    function xmDot = formation_control(m_)
        % Identify the friends of agent at-hand
        tmp_ = 1:N_AGENTS;
        friendAgents = tmp_(adjacency_(m_, :) > 0);

        % % Calculate the desired position for agent m_
        % angle = 2 * pi / N_AGENTS;							% <<<====== 
        % radius = 0.3;											% <<<====== 
        % desiredPosition = [radius * cos((m_ - 1) * angle);... % <<<====== 
        %                    radius * sin((m_ - 1) * angle)];	% <<<====== 
		% MOVED THIS TO MAIN SCRIPT TO DEFINE DESIRED **ABSOLUTE** POSITION
		% ONCE, INSTEAD OF REDEFINING AT EACH TIME STEP HERE

        % Update based on friend data
        um_ = 0;

        % Threshold
        % delta = 0.25;   % tri = 0.25 / sq = 0.25 / pen = 0.25 / hex = 0.25 % <<<====== 
		% REMOVED ABOVE LINE. MAX COMMS DISTANCE THRESHOLD MUST NOT BE SET HERE.
		% IT IS A GLOBAL VARIABLE PREVIOUSLY DEFINED.

        for m2 = friendAgents
            % dij = desiredPosition - desiredPosition(:, 1);	% <<<====== 
			% REMOVED AND UPDATED ABOVE LINE BECAUSE IT DIDN'T MAKE SENSE
			% AND DID NOT REFLECT THE QUANTITY d_ij AS DEFINED IN THE PAPER
			desiredRelativePosn	= desiredAbsolutePosn(:, m_) - desiredAbsolutePosn(:, m2);
            
			% lij = norm(x_(:, m_) - x_(:, m2));				% <<<====== 
			% UPDATED LINE ABOVE TO CALCULATE DISTANCE USING y_, INSTEAD OF x_, 
			% DUE TO WHICH WE CAN INCORPORATE COMMS DELAYS IN THE SIM 
			distanceToFriend = norm( x_(:, m_) - x_(:, m2) );

            control_term = ... % <<<====== 
				(2 * (MAX_COMMS_DISTANCE - norm(desiredRelativePosn)) - ...
				norm(distanceToFriend - desiredRelativePosn)) /...
				((MAX_COMMS_DISTANCE - norm(desiredRelativePosn) - ...
				norm(distanceToFriend - desiredRelativePosn))^2);
			% CRUCIALLY, THE CONTROL TERM WAS MISSING A NEGATIVE SIGN;
			% FIXED TO REFLECT CORRECT SIGN

            um_ = um_ - (control_term * (x_(:, m_) - x_(:, m2) - desiredRelativePosn));
        end

        xmDot = um_;
    end
end
