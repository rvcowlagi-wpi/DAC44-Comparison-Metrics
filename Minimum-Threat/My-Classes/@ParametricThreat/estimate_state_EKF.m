%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by
	Prakash Poudel
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
An implementation of the Extended Kalman Filter to estimate threat.
%}

function obj = estimate_state_EKF(obj, time_step_, measurementz_k, sensors_)
% 	estimate_state(stateEstimatex_km1, covarPxx_km1, measurementz_k, ...
% 	stateModel, measurementModel, ESTIMATOR_PARAMETERS)

%{
INPUTS
------

The "_km1" subscript indicates (k-1). This file uses the convention of
(k-1) for current and k for next.

OUTPUTS
-------
* stateEstimatex_k: new state estimate (updaated in the threat object)
* covarPxx_k: new estimtion error covariance (updaated in the threat object)

%}

t_	= obj.timeStampEstimate(end);
% dt_ = 0.01;
 dt_ =  obj.timeStampEstimate(2);

stateEstimatex_km1	= obj.stateEstimate;
covarPxx_km1		= obj.estimateCovarPxx;

%----- Noise statistics
processCovarQ	= obj.noiseCovarQ;
measNoiseCovarR	= sensors_.noiseVariance * eye(sensors_.nSensors);

%----- Linearization
F   = eye(obj.nStates) + obj.A * dt_;
G_2 = eye(obj.nStates) * dt_;
C   = eye(obj.nStates);

%----- Calculate state and covariance of the prediction. 
predx_kkm1 = obj.state;
P_kkm1     = F * covarPxx_km1 * F' + G_2 * processCovarQ * G_2';
% P_kkm1     =  obj.A * covarPxx_km1 * obj.A' + processCovarQ * dt_;	

%----- Return if no measurement available

if isempty(measurementz_k)
	obj.stateEstimate	 = stateEstimatex_km1;
	obj.estimateCovarPxx = P_kkm1;
	return
end

%----- Kalman gain (if measurement available)
KalmanGain	= P_kkm1 * C' / (C * P_kkm1 * C' + measNoiseCovarR);

%----- Correct (if measurement available)
obj.stateEstimate	=  predx_kkm1  + KalmanGain * (measurementz_k - C * predx_kkm1 );
obj.estimateCovarPxx= (eye(obj.nStates) - KalmanGain * C) * P_kkm1;

%----- Trace of error covariance
obj.traceCovarPxx = trace(obj.estimateCovarPxx);

%----- Update object histories
obj.stateEstimateHistory	= [obj.stateEstimateHistory		obj.stateEstimate];
obj.estimateCovarPxxHistory	= [obj.estimateCovarPxxHistory	reshape(obj.estimateCovarPxx, obj.nStates^2, 1)];
obj.traceCovarPxxHistory	= [obj.traceCovarPxxHistory	obj.traceCovarPxx];
obj.timeStampEstimate		= [obj.timeStampEstimate		t_ + time_step_];

	
%----------------------------------------------------------------------

end