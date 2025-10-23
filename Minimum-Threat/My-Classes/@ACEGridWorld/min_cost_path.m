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
Path optimization in the class definition of grid world. "Discovers"
neighbours and assigns vertex numbers as they are discovered.
%}

function obj = min_cost_path(obj)

%------ Initialization
idStart		= obj.searchSetup.start;

nVertices	= 1E3;														% For large graphs, initialize search data structure with arbitrary size
knownIDs	= Inf(nVertices, 1);									
% This is the list of known vertex IDs; the row number in this list is the
% vertex number for that ID 

% The assumption here is that the function that discovers neighbouring
% vertices also provides a **unique** identifier to each vertex. The main
% difference from a typical Dijkstra implementation is that the vertex ID
% (which can be large) need not match this algorithm's internal indexing,
% which is helpful when the #vertices searched << total number of vertices.
obj.searchOutcome		= [];
tmp	= struct('id', 0, 'mk', 0, 'd', Inf, 'b', [], 'x', []);
obj.searchOutcome		= repmat(tmp, 1, nVertices);

% ID 'id' and state 'x' are redundant; 'ID' is scalar

obj.searchOutcome(1).id	= idStart;
obj.searchOutcome(1).mk = 1;
obj.searchOutcome(1).d	= 0;

nKnownIDs	= 1;
knownIDs(1)	= idStart;

fringe_			= [1 0];
nFringe			= 1;

isGoalClosed	= 0;

nIter	= 0;
while (nFringe ~= 0) && (~isGoalClosed)	
	nIter		= nIter + 1;
	vCurrent	= fringe_(1, 1);											% Get vertex from top of (sorted) open stack
	obj.searchOutcome(vCurrent).mk = 2;											% Mark that vertex as dead
	
	nFringe		= nFringe - 1;
	fringe_(1, :) = [];	

	[nhbrIDs, nhbrCosts] = obj.find_neighbours(obj.searchOutcome(vCurrent).id);				% Other function handle for nhbrs and costs

% 	fprintf('\n-- Iteration: %i \n', nIter)
	
	for k = 1:numel(nhbrIDs)													% For all neighbours

		if any(nhbrIDs(k) == knownIDs(1:nKnownIDs))
			tmp2		= 1:nKnownIDs;
			vneighbour	= tmp2(nhbrIDs(k) == knownIDs(1:nKnownIDs));
			vNew		= vneighbour;
		else
			vNew		= nKnownIDs + 1;
			obj.searchOutcome(vNew).id	= nhbrIDs(k);
			obj.searchOutcome(vNew).mk	= 0;
			
			nKnownIDs		= vNew;
			knownIDs(vNew)	= nhbrIDs(k);	
		end
		costNew	= nhbrCosts(k);												% Cost to go from act to new

% 		disp([k vCurrent vNew])
% 		fprintf('\n-- Known IDs --\n')
% 		disp((knownIDs(1:nKnownIDs))')
		
		if obj.searchOutcome(vNew).mk == 0											% Unvisited
			obj.searchOutcome(vNew).mk	= 1;										% Mark open
			obj.searchOutcome(vNew).d	= obj.searchOutcome(vCurrent).d + costNew;	% Update c2come of newly visited state
			obj.searchOutcome(vNew).b	= vCurrent;			

			fringe_binary_sort([vNew obj.searchOutcome(vNew).d]);					% Add [vertexNew cost] to sorted open list
		elseif obj.searchOutcome(vNew).mk == 1										% Already open, update c2come if necessary
			if obj.searchOutcome(vNew).d > obj.searchOutcome(vCurrent).d + costNew
				obj.searchOutcome(vNew).d	= obj.searchOutcome(vCurrent).d + costNew;
				obj.searchOutcome(vNew).b	= vCurrent;
				
				fringe_( (fringe_(1:nFringe, 1) == vNew), :) = [];
				nFringe	= nFringe - 1;
				
				fringe_binary_sort([vNew obj.searchOutcome(vNew).d]);				% Add [vertexNew cost] to sorted open list
			end
		end


		
	end
	
	isGoalKnown	= (obj.searchSetup.virtualGoalID == knownIDs);
	if any(isGoalKnown) && obj.searchOutcome(isGoalKnown).mk == 2
		isGoalClosed = 1;
	end

end

knownIDs = knownIDs(1:nKnownIDs);

optimalPathIDs = obj.searchSetup.virtualGoalID;
isGoalKnown	= (obj.searchSetup.virtualGoalID == knownIDs);
tmp2		= 1:nKnownIDs;
vGoal		= tmp2(isGoalKnown);
vCurrent	= vGoal;
while (vCurrent ~= 1)
	vCurrent	= obj.searchOutcome(vCurrent).b;
	optimalPathIDs	= cat(2, knownIDs(vCurrent), optimalPathIDs);
end
% optimalPathIDs
% obj.searchOutcome(vGoal).d

optimalPathIDs(end) = [];
pathPointsInGrid = mod(optimalPathIDs, obj.nPoints);
pathPointsInGrid(pathPointsInGrid == 0) = obj.nPoints;
pathPointsInTime = floor( (optimalPathIDs - pathPointsInGrid) / obj.nPoints );

% pathPointsInGrid
% pathPointsInTime


%----- Outputs
obj.optimalPath.IDs	= optimalPathIDs;
obj.optimalPath.loc = pathPointsInGrid;
obj.optimalPath.time= pathPointsInTime;
obj.optimalPath.Cost= obj.searchOutcome(vGoal).d;
obj.pathCost		= obj.searchOutcome(vGoal).d;
obj.pathRisk		= 0;


	%======================================================================
	function fringe_binary_sort(B)
		% The rows of B are inserted into A, while sorting (ascending)
		% according to column c. Both A and B have the same number of
		% columns. A is assumed to sorted ascending.

		A	= fringe_(1:nFringe, :);
		c	= 2;
		
		[rA, cA] = size(A);
		[rB, cB] = size(B);
		
		if numel(A) == 0
			nFringe	= size(B, 1);
			fringe_ = B;
			return;
		end
		
		if numel(B) == 0, return; end
		if cB ~= cA, error('A and B must have same number of columns!\n'); end
		
		for m22 = 1:rB
			thisIns		= B(m22, :);
			thisCost	= thisIns(1, c);
			
			if ismember(thisIns, A, 'rows')
				fprintf('This one came back!\t\t'); disp(thisIns)
				redn = redn + 1;
				continue;
			end
		
			if A(rA, c) <= thisCost											% Insert after last row
				A	= cat(1, A, thisIns);
				rA	= rA + 1;
				continue;
			elseif A(1, c) >= thisCost										% Insert before first row
				A	= cat(1, thisIns, A);
				rA	= rA + 1;
				continue;
			end
			
			nCand	= rA;													% Number of candidate rows in A that can have greater cost
			testRow	= 0;
			dirn	= 1;	
			while nCand > 1
				p		= floor(nCand/2);
				testRow = testRow + dirn*p;
				dirn	= 2*(A(testRow, c) < thisCost) - 1;
				nCand	= nCand - p;
			end	
		
			insRow = testRow + (dirn + 1)/2;								% Insert before this row in A
			A	= cat(1, A(1:(insRow - 1), :), thisIns, A(insRow:rA, :));
			rA	= rA + 1;
		end

		nFringe					= size(A, 1);
		fringe_(1:nFringe, :)	= A;
	end


end

