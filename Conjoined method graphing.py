import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Lagrangian_Chaotic_Pendulum import x1 as x1_lag, y1 as y1_lag, x2 as x2_lag, y2 as y2_lag, y as y_lag, L1 as L1_lag, L2 as L2_lag, dt as dt_lag
from RK4_Chaotic_Pendulum import y as y_rk, L1 as L1_rk, L2 as L2_rk, dt as dt_rk

from numpy import sin, cos

#Calculate RK4 positions
x1_rk = L1_rk*sin(y_rk[:,0])
y1_rk = -L1_rk*cos(y_rk[:,0])
x2_rk = L2_rk*sin(y_rk[:,2]) + x1_rk
y2_rk = -L2_rk*cos(y_rk[:,2]) + y1_rk

#set up side by side animation
L = L1_lag + L2_lag
fig, (ax1, ax2)= plt.subplots(1,2, figsize=(10,5))

#lagrangian subplot
ax1.set_xlim(-L,L)
ax1.set_ylim(-L, 1.)
ax1.set_aspect('equal')
ax1.grid()
ax1.set_title('Lagrangian')

line1, = ax1.plot([],[],'-o', lw=2, color='deeppink', zorder = 3)
trace1, = ax1.plot([], [], '-o', lw=1, ms=2, color='pink', alpha=0.5, zorder = 1)
time_text1 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

#RK4 subplot
ax2.set_xlim(-L,L)
ax2.set_ylim(-L, 1.)
ax2.set_aspect('equal')
ax2.grid()
ax2.set_title('RK4')

line2, = ax2.plot([],[],'-o', lw=2, color='seagreen', zorder = 3)
trace2, = ax2.plot([], [], '-o', lw=1, ms=2, color='lawngreen', alpha=0.5, zorder = 1)
time_text2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

def animate(i):
    #Lagrangian
    thisx1 = [0, x1_lag[i], x2_lag[i]]
    thisy1 = [0, y1_lag[i], y2_lag[i]]
    line1.set_data(thisx1, thisy1)
    trace1.set_data(x2_lag[:i], y2_lag[:i])
    time_text1.set_text(f'time = {i*dt_lag:.1f}s')

    #RK4
    thisx2 = [0, x1_rk[i], x2_rk[i]]
    thisy2 = [0, y1_rk[i], y2_rk[i]]
    line2.set_data(thisx2, thisy2)
    trace2.set_data(x2_rk[:i], y2_rk[:i])
    time_text2.set_text(f'time = {i*dt_rk:.1f}s')

    return line1, trace1, time_text1, line2, trace2, time_text2

ani = animation.FuncAnimation(
    fig, animate, len(y_lag), interval = dt_lag*1000, blit=True)

plt.tight_layout()
plt.show()
