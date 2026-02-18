"""
Code sourced from: https://scipython.com/blog/the-double-pendulum/

air resistance and bearing friction damping coefficients added to increase realism
animation added for personal preference
internal ODE solver swapped for Euler to negate adaptive smoothing of chaos
this code will be used in conjunction with an RK$ code externally sourced to compare
numerical method solvers to a real double pendulum
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib
matplotlib.use('TkAgg')

import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 0.245, 0.2
m1, m2 = 0.3553, 0.2379
# The gravitational acceleration (m.s-2).
g = 9.81

# Damping coefficients
b1, b2 = 0.1, 0.1   # bearing friction coefficient on pendulum arms
c1, c2 = 0.05, 0.05 # air resistance coefficient

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    z1dot -= b1*z1 + c1 * z1 * abs(z1)  #add damping forces

    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    z2dot -= b2*z2 + c2 * z2 * abs(z2)  #add damping forces
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 30, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
th1 = 89
th2 = 89
y0 = np.array([np.radians(th1), 0, np.radians(th2), 0])

# Do the numerical integration of the equations of motion using Euler to avoid adaptive smoothing
y = np.empty((len(t), 4))
y[0] = y0
for i in range (1, len(t)):
    y[i] = y[i-1] + np.array(deriv(y[i-1], t[i-1], L1, L2, m1, m2)) * dt


# Unpack z and theta as a function of time
theta1, theta2 = y[:,0], y[:,2]

# Convert to Cartesian coordinates of the two bob positions.
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

########################################
#CHANGING THE CODE TO DISPLAY ANIMATION
########################################

# Set up animation
# L = L1 + L2
# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(autoscale_on=False, xlim=(-L,L), ylim=(-L,1.))
# ax.set_aspect('equal')
# ax.grid()
#
# line, = ax.plot([],[], '-o', lw=2, zorder = 3)
# trace, = ax.plot([], [], '-o', lw=1, ms=2, zorder = 1)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#
# def animate(i):
#     thisx = [0, x1[i], x2[i]]
#     thisy = [0, y1[i], y2[i]]
#
#     history_x = x2[:i]
#     history_y = y2[:i]
#
#     line.set_data(thisx, thisy)
#     trace.set_data(history_x, history_y)
#     time_text.set_text(time_template % (i*dt))
#     return line, trace, time_text
#
# ani = animation.FuncAnimation(
#
#     fig, animate, len(y), interval = dt*1000, blit=True)
# plt.show()
