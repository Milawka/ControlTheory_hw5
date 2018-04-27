# ControlTheory_hw5
======
###### To run my code you probably need:

⋅⋅⋅sudo apt install python-pip
⋅⋅⋅pip install python-tk
⋅⋅⋅pip install numpy matplotlib
⋅⋅⋅pip install filterpy --user
 
and finally:
⋅⋅⋅python m.py 
------

#    REPORT
0. I installed all needed pacjages
1. I wrote a code
  ⋅⋅1. First of all I tryed to see what trajectory I will get usisng data from  camera. I wrote code for this case. I saw a plot. (blue line)
  ⋅⋅2. Next I try to build this plot for if we had ideal model (without noise), using formulas from pdf you gave us, from chapter "Forward Kinematics for Differential Drive Robot". Using data taken from camera and gyroscope. I get a plot and see it.(orange line)
  ⋅⋅3. And finally I try to build a plot from data taken from the camera and gyroscope and applying to this kalman filter. And for this case I used a library (for Kalman) (green line)
  
ps: I also swaped Vl and Vr
