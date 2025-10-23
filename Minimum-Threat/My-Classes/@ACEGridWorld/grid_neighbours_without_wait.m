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
4-adjacency grid neighbours on a uniform grid. Time is updated but there
are no "waiting" neighbours.
%}

function [nhbrIDs, nhbrCosts] = grid_neighbours_without_wait(obj, currentID)


nhbrIDs		= [];
nhbrCosts	= [];


% ID = number of spatial grid points * time samples elapsed + current grid
% point number

pointInGrid = mod(currentID, obj.nPoints);
if pointInGrid == 0, pointInGrid = obj.nPoints; end
pointinTime = floor( (currentID - pointInGrid) / obj.nPoints );

if mod( pointInGrid, obj.nGridRow )
	% pointInGrid + 1 is a neighbour
	newNeighbour= (pointInGrid + 1); % + obj.nPoints * (pointinTime + 1);
	newCost		= obj.threatModel.calculate_at_locations(...
		obj.coordinates(:, newNeighbour), obj.threatModel.stateHistory(:, 1));
	nhbrIDs		= [nhbrIDs; newNeighbour];
	nhbrCosts	= [nhbrCosts; newCost];
end
if mod( pointInGrid - 1, obj.nGridRow )
	% pointInGrid - 1 is a neighbour
	newNeighbour= (pointInGrid - 1); % + obj.nPoints * (pointinTime + 1);
	newCost		= obj.threatModel.calculate_at_locations(...
		obj.coordinates(:, newNeighbour), obj.threatModel.stateHistory(:, 1));
	nhbrIDs		= [nhbrIDs; newNeighbour];
	nhbrCosts	= [nhbrCosts; newCost];
end

if pointInGrid + obj.nGridRow <= obj.nPoints
	% pointInGrid + obj.nGridRow is a neighbour
	newNeighbour= (pointInGrid + obj.nGridRow); % + obj.nPoints * (pointinTime + 1);
	newCost		= obj.threatModel.calculate_at_locations(...
		obj.coordinates(:, newNeighbour), obj.threatModel.stateHistory(:, 1));
	nhbrIDs		= [nhbrIDs; newNeighbour];
	nhbrCosts	= [nhbrCosts; newCost];
end

if pointInGrid - obj.nGridRow >= 1
	% pointInGrid - obj.nGridRow is a neighbour
	newNeighbour= (pointInGrid - obj.nGridRow); % + obj.nPoints * (pointinTime + 1);
	newCost		= obj.threatModel.calculate_at_locations(...
		obj.coordinates(:, newNeighbour), obj.threatModel.stateHistory(:, 1));
	nhbrIDs		= [nhbrIDs; newNeighbour];
	nhbrCosts	= [nhbrCosts; newCost];
end

if pointInGrid == obj.searchSetup.locationGoal
	newNeighbour= obj.searchSetup.virtualGoalID;
	newCost		= 0;
	nhbrIDs		= [nhbrIDs; newNeighbour];
	nhbrCosts	= [nhbrCosts; newCost];
end


end