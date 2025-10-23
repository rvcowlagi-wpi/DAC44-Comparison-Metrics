%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by Raghvendra V. Cowlagi

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
This script generates trajectory examples of an uncontrolled dynamical
system.
%}

clear variables; close all; clc;

%% Problem setup
nTrials		= 1000;
nTimePts	= 1000;
tFin		= 10;
nState		= 6;
dataSize	= nTimePts*nState;
case_		= 1;
set_		= 9;


%% Book-keeping
caseName	= ['case' num2str(case_, '%5.2i')];
setName		= ['set' num2str(set_, '%5.2i')];
caseHandle	= str2func(caseName);

foldername_ = ['Data/' caseName '/' setName '/'];
if ~exist(foldername_, 'dir')
	mkdir(foldername_)
else
	delete([foldername_ '/*.csv'])
	delete([foldername_ '/*.mat'])
end
systemDataFileName = [foldername_ 'systemData.mat'];

%% Stable linear system with randomly chosen eigenvalues
nComplexPair= floor(nState/4);
nReal		= nState - 2*nComplexPair;

rng(0.9*set_, 'twister');
realEVs		= -5 + 4*rand(nReal, 1);
complexPart = 5*rand(nComplexPair, 1);
realPart	= -5 + 4*rand(nComplexPair, 1);

%----- Make a temporary A  matrix in block form
A_ = zeros(nState);
for m1 = 1:nReal
	A_(m1, m1) = realEVs(m1);
end
for m1 = 1:nComplexPair
	m2 = nReal + 2*(m1 - 1) + 1;
	p_ = poly([realPart(m1) + complexPart(m1)*1i; realPart(m1) - complexPart(m1)*1i]);
	A_( m2:m2+1, m2:m2+1 ) = [0 1; -p_(3) -p_(2)];
end

%----- Random symmetric positive definite matrix
S	= sprandsym(nState, 1);

%----- Get A matrix from a similarity transformation 
A	= S * A_ / S;

C	= [1 zeros(1, nState-1)];

%----- Noise transformation
G	= 0.5*randn(nState, 1);

systemParameters.A = A;
systemParameters.G = G;

% save(systemDataFileName, 'A', 'G')

ySimStore = zeros(nTrials, nTimePts + 1);

%% Run trials
figure
for m = 1:nTrials
	xSim	= single(caseHandle(m, nState, nTimePts, tFin, systemParameters));
	ySim	= C*xSim;
	ySimStore(m, :)	= ySim;
	filename_	= [foldername_ 'traj_' num2str(m, '%5.4i') '.csv'];

	% figure
	plot(linspace(0,tFin, nTimePts+1), ySim, 'LineWidth', 2); hold on

	% writematrix(xSim, filename_);

end

%% Scatter plot

yMean	= mean(ySimStore, 1);
yData	= ySimStore;

[pcaU_, pcaS_, ~] = svd(yData, "econ");
yPlot = pcaU_(:, 1:3) * pcaS_(1:3, 1:3);

figure;
plot3(yPlot(:, 1), yPlot(:, 2), yPlot(:, 3), '.')
