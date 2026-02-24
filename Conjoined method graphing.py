import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.interpolate import interp1d
import pandas as pd

from Lagrangian_RK4 import x1 as x1_lag, y1 as y1_lag, x2 as x2_lag, y2 as y2_lag, y as y_lag, L1 as L1_lag, L2 as L2_lag, dt as dt_lag
from Newtonian_RK4 import y as y_rk, L1 as L1_rk, L2 as L2_rk, dt as dt_rk

from numpy import sin, cos

#Calculate RK4 positions
x1_rk = L1_rk*sin(y_rk[:,0])
y1_rk = -L1_rk*cos(y_rk[:,0])
x2_rk = L2_rk*sin(y_rk[:,2]) + x1_rk
y2_rk = -L2_rk*cos(y_rk[:,2]) + y1_rk

def sync_experimental_data(sim_time_array, real_timestamps, real_x1, real_y1, real_x2, real_y2):
    """Interpolates real-world data to match the simulation's time steps."""
    # Create interpolation functions for each coordinate
    fx1 = interp1d(real_timestamps, real_x1, bounds_error=False, fill_value="extrapolate")
    fy1 = interp1d(real_timestamps, real_y1, bounds_error=False, fill_value="extrapolate")
    fx2 = interp1d(real_timestamps, real_x2, bounds_error=False, fill_value="extrapolate")
    fy2 = interp1d(real_timestamps, real_y2, bounds_error=False, fill_value="extrapolate")

    # Map the real data onto the simulation's timeline
    return fx1(sim_time_array), fy1(sim_time_array), fx2(sim_time_array), fy2(sim_time_array)

# --- Real Data Loading and Syncing ---
# Replace 'data.csv' with your actual filename and check column names
df_real = pd.read_csv('formatted_pendulum_data.csv')
t_raw = df_real['time'].values
x1_raw, y1_raw = df_real['x1'].values, df_real['y1'].values
x2_raw, y2_raw = df_real['x2'].values, df_real['y2'].values

# Create the simulation time array
sim_time = np.arange(0, len(y_lag) * dt_lag, dt_lag)

# Sync the real data to the sim time
x1_real, y1_real, x2_real, y2_real = sync_experimental_data(
    sim_time, t_raw, x1_raw, y1_raw, x2_raw, y2_raw
)

#set up side by side animation
L = L1_lag + L2_lag
#fig, (ax1, ax2)= plt.subplots(1,2, figsize=(10,5))

# Update to 1 row, 3 columns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# ... (Keep ax1 and ax2 code as is) ...

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

# Setup ax3 (Real Pendulum)
ax3.set_xlim(-L, L)
ax3.set_ylim(-L, 1.)
ax3.set_aspect('equal')
ax3.grid()
ax3.set_title('Real World Data')

line2, = ax2.plot([],[],'-o', lw=2, color='seagreen', zorder = 3)
trace2, = ax2.plot([], [], '-o', lw=1, ms=2, color='lawngreen', alpha=0.5, zorder = 1)
time_text2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)
line3, = ax3.plot([], [], '-o', lw=2, color='royalblue')
trace3, = ax3.plot([], [], '-', lw=1, color='cornflowerblue', alpha=0.5)
time_text3 = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

# OLD ANIMATE FUNCTION
# def animate(i):
#     #Lagrangian
#     thisx1 = [0, x1_lag[i], x2_lag[i]]
#     thisy1 = [0, y1_lag[i], y2_lag[i]]
#     line1.set_data(thisx1, thisy1)
#     trace1.set_data(x2_lag[:i], y2_lag[:i])
#     time_text1.set_text(f'time = {i*dt_lag:.1f}s')
#
#     #RK4
#     thisx2 = [0, x1_rk[i], x2_rk[i]]
#     thisy2 = [0, y1_rk[i], y2_rk[i]]
#     line2.set_data(thisx2, thisy2)
#     trace2.set_data(x2_rk[:i], y2_rk[:i])
#     time_text2.set_text(f'time = {i*dt_rk:.1f}s')
#
#     return line1, trace1, time_text1, line2, trace2, time_text2
#
# ani = animation.FuncAnimation(
#     fig, animate, len(y_lag), interval = dt_lag*1000, blit=True)

def animate(i):
    # Lagrangian
    line1.set_data([0, x1_lag[i], x2_lag[i]], [0, y1_lag[i], y2_lag[i]])
    trace1.set_data(x2_lag[:i], y2_lag[:i])
    time_text1.set_text(f'time = {i*dt_lag:.1f}s')

    # RK4
    line2.set_data([0, x1_rk[i], x2_rk[i]], [0, y1_rk[i], y2_rk[i]])
    trace2.set_data(x2_rk[:i], y2_rk[:i])
    time_text2.set_text(f'time = {i*dt_rk:.1f}s')

    # Real
    line3.set_data([0, x1_real[i], x2_real[i]], [0, y1_real[i], y2_real[i]])
    trace3.set_data(x2_real[:i], y2_real[:i])
    time_text3.set_text(f'time = {i*dt_lag:.1f}s')

    return line1, trace1, time_text1, line2, trace2, time_text2, line3, trace3, time_text3

# Note: Ensure blit=True is still there for performance
ani = animation.FuncAnimation(
    fig, animate, frames=len(y_lag), interval=dt_lag*1000, blit=True)

plt.tight_layout()
plt.show()
