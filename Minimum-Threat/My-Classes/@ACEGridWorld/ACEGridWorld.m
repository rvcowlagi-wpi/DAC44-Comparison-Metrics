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
Class definition of grid world, including optimal (minimum threat) planning.
	* Uniform spacing
	* 4-adjacency
	* Works with ParametricThreat class to get threat costs
%}

classdef ACEGridWorld
	properties
		halfWorkspaceSize	% half of square workspace edge size
		nPoints				% number of grid points in xy space, t is different
		nGridRow			% number of grid points in each xy row
		spacing				% xy spacing

		coordinates			% xy coordinates, t is different
		adjacency			% xy adjacency matrix

		optimalPath
		pathCost
		pathRisk

		searchSetup
		searchOutcome		% label, backpointer, etc
		
		threatModel
	end

	methods
		%==================================================================
		function obj = ACEGridWorld(halfWorkspaceSize_, nGridRow_)
			% Initialization

			obj.halfWorkspaceSize	= halfWorkspaceSize_;
			obj.nGridRow			= nGridRow_;

			obj.nPoints				= nGridRow_ ^ 2;
			obj.spacing				= 2*halfWorkspaceSize_ / (nGridRow_ - 1);

			obj.coordinates	= zeros(2, obj.nPoints);
			for m1 = 0:(obj.nPoints - 1)	
				obj.coordinates(:, m1 + 1) = [...
					-halfWorkspaceSize_ + (mod(m1, nGridRow_)) * obj.spacing; ...
					-halfWorkspaceSize_ + floor(m1 / nGridRow_) * obj.spacing];
			end

			% Setup adjacency matrix
			nEdges		= 0;
			nExpEdges	= obj.nPoints * 4;
			edgeList	= zeros(nExpEdges, 3);
			for m1 = 1:obj.nPoints
				if (m1 + 1 <= obj.nPoints) && (mod(m1, nGridRow_) ~= 0)
					nEdges				= nEdges + 1;
					edgeList(nEdges, :) = [m1 (m1 + 1) 1];
					nEdges				= nEdges + 1;
					edgeList(nEdges, :) = [(m1 + 1) m1 1];
				end
			
				if (m1 + nGridRow_) <= obj.nPoints
					nEdges				= nEdges + 1;
					edgeList(nEdges, :) = [m1 (m1 + nGridRow_) 1];
					nEdges				= nEdges + 1;
					edgeList(nEdges, :) = [(m1 + nGridRow_) m1 1];
				end
			end
			obj.adjacency = sparse(edgeList(1:nEdges, 1), ...
				edgeList(1:nEdges, 2), edgeList(1:nEdges, 3));

			obj.optimalPath = [];
			obj.pathCost	= Inf;
			obj.pathRisk	= Inf;

			obj.searchSetup.start			= [];
			obj.searchSetup.locationGoal	= obj.nPoints;
			obj.searchSetup.virtualGoalID	= 0;

			obj.threatModel = [];
		end
		%------------------------------------------------------------------

		%==================================================================
		obj = min_cost_path(obj)
		% Path optimization function in a separate file
		%------------------------------------------------------------------

		%==================================================================
		function [nhbrIDs, nhbrCosts] = find_neighbours(obj, currentID)
			[nhbrIDs, nhbrCosts] = grid_neighbours_without_wait(obj, currentID);
		end
		% Neighbour discovery function in a separate file
		%------------------------------------------------------------------

		%==================================================================
		isGoalClosed = goal_check_locationanytime(obj)
		% Neighbour discovery function in a separate file
		%------------------------------------------------------------------

		%==================================================================
		plot_(obj, threat_, sensors_, threatState_, sensorConfig_, planState_, flags_)
		% Path optimization function in a separate file
		%------------------------------------------------------------------
	end
end