
import matplotlib
matplotlib.use('svg')

import numpy as np
import matplotlib.pyplot as plt
import math


plt.figure()

import csv
cs = []
with open('log_camera_2.csv','rb') as f:
	reader = csv.reader(f, delimiter=';')
	reader.next()
	for _t, _x, _y in reader:
		cs.append(map(float, (_t, _x, _y)))

from scipy.interpolate import interp1d
_t, _x, _y = zip(*cs)
camX = interp1d(_t, _x)
camY = interp1d(_t, _y)
camMaxT = _t[-1]

if 1:
	x, y = [], []
	for _, _x, _y in cs:
		x.append(_x)
		y.append(_y)
	plt.plot(x, y)

def step((x, y, a), dt, vl, vr):
	vl, vr = vr, vl
	if vr-vl:
		l = 15.0
		R = (l*(vl+vr))/(2*(vr-vl))
		w = (vr - vl) / l
		ICCx, ICCy = [x - R*math.sin(a), y+R*math.cos(a)]

	
		return np.dot(np.array([[math.cos(w*dt), -math.sin(w*dt), 0], [math.sin(w*dt), math.cos(w*dt), 0], [0, 0, 1]]), np.array([[x-ICCx, y-ICCy, a]]).T) + np.array([[ICCx, ICCy, w*dt]]).T
	return np.array([[x + dt * vl * math.cos(a), y + dt * vl * math.sin(a), a]]).T

import csv
with open('log_robot_2.csv','rb') as f:
	reader = csv.reader(f, delimiter=';')
	reader.next()
	s = []
	for _t, _, _g, _vl, _vr in reader:
		s.append(map(float, (_t, _g, _vl, _vr)))
f = lambda v: 2 * np.pi * 2.7 * v / 360
if 1:
	x, y = [], []
	X = [[0.0, 0.0, 0.0]]
	for i in range(len(s) - 1):
		t, _, vl, vr = s[i]
		if t > camMaxT:
			break
		#import pdb; pdb.set_trace()
		X = step(X[0], s[i+1][0] - t, f(vl), f(vr)).T
		x.append(X[0][0])
		y.append(X[0][1])
	plt.plot(x, y)

#kalman
x, y = [], []

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
my_filter = KalmanFilter(dim_x=3, dim_z=3)
 # initial state (location and velocity)
my_filter.x = np.array([[0.0, 0.0, 0.0]]).T      
 # state transition matrix
my_filter.F = np.eye(3)   
# Measurement function
my_filter.H = np.eye(3)  
 # covariance matrix  
my_filter.P *= 20.0
# state uncertainty                
my_filter.R = np.diag([49.0, 49.0, math.radians(16.0)])                      
for i in range(len(s) - 1):
	t, g, vl, vr = s[i]
	g = math.radians(g)
	if t > camMaxT:
		break
	my_filter.x = step(my_filter.x.T[0], s[i+1][0] - t, f(vl), f(vr))
	my_filter.update(np.array([camX(t), camY(t), g]).T)
	x.append(my_filter.x.T[0][0])
	y.append(my_filter.x.T[0][1])
plt.plot(x, y)


plt.gca().set_position([0, 0, 1, 1])
plt.axis('equal')
plt.savefig("test.svg",bbox_inches='tight')







