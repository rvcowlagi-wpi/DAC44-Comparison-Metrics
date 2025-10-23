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
Check if goal is closed: this function checks for goal a spatial
destination to be reached at any time.
%}

function isGoalClosed = goal_check_locationanytime(obj)



if ~isAllGoalSearch
		for k = 1:numel(idGoal)
			isGoalKnown		= (idGoal(k) == knownVertices);
			if any(isGoalKnown) && vertexData(isGoalKnown).mk == 2
				isGoalClosed = 1;
				break;
			end
		end
	else			
% 		if ~all(ismember(idGoal, knownVertices)), continue; end
% 		isGoalClosed = 1;
% 		for k = 1:numel(idGoal)
% 			[~, idx_goal] = ismember(idGoal(k), knownVertices);
% 			if vertexData(idx_goal).mk == 2, isGoalClosed = 0; break; end
% 		end