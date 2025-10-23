%{
SOFTWARE LICENSE
----------------
Copyright (c) 2021 by Raghvendra V. Cowlagi

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
Generates a subset from a larger dataset of Zermelo trajectories.
%}

%% Load large dataset
% Don't change anything in this section

clear variables; close all; clc;

load dataset2_var_wind.mat baseline_data
nMaxTrajectories	= size(baseline_data, 2);
nMaxDiscretization	= (size(baseline_data, 1) - 7)/8;

%% Choose subset parameters

nTrajectories	= 2000;		% Choose how many trajectories wanted in data-subset
nDiscretization	= 25;		% Choose number of discretization points
nStates			= 7;		% Choose how many of the 7 states to keep
% The states are in the order (x1, x2, psi, p1, p2, wind1, wind2)
% This selection will keep the first "nStates" states, i.e.,
% if nStates = 3, the subset will keep (x, y, psi)

%% Sanity checks and bookkeeping folder names etc.
% Don't change anything in this section

nTrajectories	= max(1, min(nTrajectories, nMaxTrajectories));
nDiscretization	= min(nDiscretization, nMaxDiscretization);			% Baseline dataset has 100 features per state
nStates			= max(2, min(nStates, 7));								% Keep at least (x1, x2)
nFeatures		= nDiscretization*nStates;

foldername_ = ['../Data/Variable-Wind-Constant-Endpoints/F' num2str(nFeatures)];
if isfolder(foldername_)
	allCSVFilesInFolder = [foldername_ '/*.csv'];
	delete(allCSVFilesInFolder)
else
	mkdir(foldername_)
end

%% Write data subset to CSV files
nSkip	= floor( nMaxDiscretization / nDiscretization );
for k1 = 1:nTrajectories

	traj_k1	= baseline_data( (8 + nMaxDiscretization) ...
		: nSkip : (7 + (nStates + 1)*nMaxDiscretization), k1 );
	filename_ = [foldername_ '/mintime_points' num2str(k1) '.csv'];
	writematrix(traj_k1, filename_ );

end



