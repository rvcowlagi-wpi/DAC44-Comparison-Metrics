%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by 
	Raghvendra V Cowlagi
	Bejamin Cooper
	Prakash Poudel

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
Class definition of parametric threat.
%}

classdef ParametricThreat
	properties
		offset
		alfa
		nStates				% Number of parameters (states)
		basisCenter
		basisSpread

		% For a time-varying threat, the threatState property should be updated
		% at every time step by the "dynamics_discrete" method 
		state

		stateEstimate		% mean state estimate
		estimateCovarPxx	% estimation error covariance
        traceCovarPxx       % trace of error covariance
        pNext                       % covariance prediction for next step
       
		% Maintain histories of state and stateEstimate evolution
		stateHistory
		stateEstimateHistory		% mean state estimate
		estimateCovarPxxHistory		% estimation error covariance
        traceCovarPxxHistory        % trace of error covariance
              
		% Maintain time stamps of state and estimate
		timeStampState
		timeStampEstimate

		% Process noise
		noiseCovarQ

		% Link to a grid world
		ACEGridWorld_

        % State transition matrix
        A
       
	end

	methods
		%==================================================================
		function obj = ParametricThreat(nStates_, halfWorkspaceSize_, ...
				sensorNoiseVar_, grid_)
			% Initialization

			obj.nStates		= nStates_;

			% basis variance chosen arbitrarily
			obj.basisCenter	= zeros(2, obj.nStates);
			for m1 = 1:obj.nStates
				obj.basisCenter(:, m1) = halfWorkspaceSize_*(-2 + 4*rand(2, 1) );
			end

			% basis variance chosen such that the range of influence of each basis just
			% touch at the diagonal between two basis centers. Basis centers are 1.4 *
			% center_spacing distance along diagonal, therefore meet at 0.7 * center
			% Range of influce (x_rng) is governed by threat value and noise level. It
			% has similar meaning/use to a signal-to-noise ratio (SNR)
			
			basisSpread_	= (0.1 + 0.5*halfWorkspaceSize_*rand(obj.nStates, 1).^2);
			obj.basisSpread	= basisSpread_;		% This is \sigma^2_\Psi
			obj.offset		= 0;

% 			exampleState	= [ 0    1    0; ...           % How the threats look visually
% 								1    5    1; ...           % but, need to flip, transpose
% 								0    1    0];
			exampleState	= 0.5 + 1.5*rand(obj.nStates, 1);
			obj.state		= exampleState; %reshape(flipud(exampleState)', [obj.nStates, 1]);
			obj.alfa		= 1e-3;


			obj.noiseCovarQ = 0.01*diag( ones(nStates_, 1) );
			obj.ACEGridWorld_= grid_;

			obj.stateEstimate	 = zeros(nStates_, 1);
			obj.estimateCovarPxx = eye(nStates_);
            obj.traceCovarPxx    = zeros(1,1);

			obj.stateHistory			 = obj.state;
			obj.stateEstimateHistory	 = obj.stateEstimate;
			obj.estimateCovarPxxHistory  = reshape(obj.estimateCovarPxx, nStates_^2, 1);
            obj.traceCovarPxxHistory     = obj.traceCovarPxx;

			obj.timeStampState			= 0;
			obj.timeStampEstimate		= 0;
            obj.pNext                   = 1.0075 * eye(nStates_,nStates_);
%             obj.observationMatrix       = zeros(1,nStates_);
		end
		%------------------------------------------------------------------

		%==================================================================
		function c_ = calculate_at_locations(obj, locations_, thisState)
			% "locations_" is either:
			%	2 x n vector, where each column has 2D position coordinates
			%	OR
			%	n x n x 2 array with meshgrid locations for 2D position	coords

			if ~exist("thisState", "var")
				thisState = obj.state;
			end

			observationH= obj.calc_rbf_value(locations_);
			c_			= obj.offset +  observationH * thisState;

			if size(locations_, 3) > 1
				c_		= reshape(c_, size(locations_, 1), size(locations_, 2));
			end

		end
		%------------------------------------------------------------------

		%==================================================================
		function obj = dynamics_discrete(obj, time_step_)

			% This will update the internal state and history. If you just
			% need a prediction for the next time step without changing the
			% state stored in this object, use "process_model" instead.

			t_	= obj.timeStampState(end);

			Phi = obj.calc_rbf_value(obj.basisCenter);
			Ac	= obj.alfa * Phi' ./ (norm(Phi)^2) * ...
				obj.calc_laplacian(obj.basisCenter);
			obj.A	= eye(obj.nStates) + Ac * time_step_;
			
			obj.state = obj.A*obj.state;

			obj.stateHistory	= [obj.stateHistory		obj.state];
			obj.timeStampState	= [obj.timeStampState	t_ + time_step_];
		end
		%------------------------------------------------------------------

		%==================================================================
		function nextState_ = process_model(obj, ...
				threatStatex_, processNoise_, time_step_)
			
			Phi = obj.calc_rbf_value(obj.basisCenter);
			Ac	= obj.alfa * Phi' ./ (norm(Phi)^2) * ...
				obj.calc_laplacian(obj.basisCenter);
			obj.A	= eye(obj.nStates) + Ac * time_step_;						% \theta_dot = A\theta

			nextState_	= obj.A*threatStatex_ + processNoise_;					% Then add noise
		end
		%------------------------------------------------------------------

		%==================================================================
		function c_ = measurement_model(obj, threatStatex_, measNoise_, sensors_)
			% Calculate at locations, then add noise

			locations_	 = obj.ACEGridWorld_.coordinates(:, sensors_.configuration);
			
			observationH = obj.calc_rbf_value(locations_);
			c_			 = obj.offset +  observationH * threatStatex_ + measNoise_;
		end
		%------------------------------------------------------------------
        
		%==================================================================
		function observationH_ = calc_rbf_value(obj, locations_)

			% "locations_" is either:
			%	2 x n vector, where each column has 2D position coordinates
			%	OR
			%	n x n x 2 array with meshgrid locations for 2D position	coords
            
			if size(locations_, 3) > 1
				nLocations		= numel(locations_(:, :, 1));
				tmpX			= locations_(:, :, 1);
				tmpY			= locations_(:, :, 2);
				locationsFlatnd = [tmpX(:) tmpY(:)]';
			else
				nLocations		= size(locations_, 2);
				locationsFlatnd	= locations_;
			end
			observationH_		= zeros(nLocations, obj.nStates);

			for m1 = 1:obj.nStates
				relLocation = locationsFlatnd - obj.basisCenter(:, m1);
				relRadiusSq = relLocation(1, :).^2 + relLocation(2, :).^2;
				observationH_(:, m1) = ...
					exp( (-1 / (2 * obj.basisSpread(m1))) .* relRadiusSq );
			end
		end
		%------------------------------------------------------------------

		%==================================================================
		function laplacianL_ = calc_laplacian(obj, locations_)
			% Laplacian (d^2/dx^2 + d^2/dy^2)(Gauss basis)

			nLocations	= size(locations_, 2);
			laplacianL_ = zeros(nLocations, obj.nStates);

			for m1 = 1:size(locations_, 2)
				locationVec_ = [locations_(1, m1)*ones(1, obj.nStates); ...
					locations_(2, m1)*ones(1, obj.nStates)];
		
				laplacianL_(m1, :) = (1 / obj.basisSpread) .* ...
					(-2 + 1/(2 * (obj.basisSpread)^2) .* ...
					( (locationVec_(1, :) - obj.basisCenter(1, :)).^2 + ...
					(locationVec_(2, :) -  obj.basisCenter(2, :)).^2))...
        			.* exp((-1 / (2 * (obj.basisSpread)^2)) .* ...
					((locationVec_(1, :) - obj.basisCenter(1, :)).^2 + ...
					(locationVec_(2, :) -  obj.basisCenter(2, :)).^2) );
			end
		end
		%------------------------------------------------------------------

		%==================================================================
		obj = estimate_state_UKF(obj, time_step_, measurementz_k, sensors_)
		% Estimator in a separate file
		%------------------------------------------------------------------

       

        %==================================================================
		obj = estimate_state_UKF1(obj, time_step_, measurementz_k, sensors_)
		% Estimator in a separate file
		%------------------------------------------------------------------

		%==================================================================
		obj = plot_(obj, flags_)
		% State and estimate plots in a different file
		%------------------------------------------------------------------
	end
end