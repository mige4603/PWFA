# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:25:16 2019

@author: michael
"""

import sys
sys.path.append('/home/michael/Documents/PWFA/Decay_Model/')

import global_variables as var
import shared_var as svar

import h5py as h5
import numpy as np
import scipy.interpolate as sci
from matplotlib import pyplot as plt

def radial_vel(vel, y, hlf=0):
    if y < hlf:
        vel = - vel

    return vel

myFile = h5.File('Data/'+svar.name+'.h5', 'r')

vel = myFile[svar.fluid+'/Velocity/time_0'][:]

vel_x_ind = int( .5 * vel.shape[0] )

y_dom_og = var.X_dom_1D * svar.s_scl * 1e3
z_dom_og = var.Z_dom_1D * svar.s_scl

pts = 501
half_pts = int(.5*pts)

y_dom = np.linspace(y_dom_og[0], y_dom_og[-1], pts) 
z_dom = np.linspace(z_dom_og[0], z_dom_og[-1], pts)

vel_rad_ln = np.empty((svar.dumps, pts))
vel_rad = np.empty((svar.dumps, pts, pts))
for d in range(svar.dumps):
    vel = svar.v_scl * myFile['{0}/Velocity/time_{1}'.format(svar.fluid, d)][vel_x_ind, 0::, 0::]
    vel_spline_y = sci.RectBivariateSpline(y_dom_og, z_dom_og, vel[0::, 0::, 1])
    for y_ind, y in enumerate(y_dom):
        for z_ind, z in enumerate(z_dom):
            vel_y = vel_spline_y(y, z)[0][0]
            vel_r = radial_vel(vel_y, y, hlf=y_dom[half_pts])
            vel_rad[d, y_ind, z_ind] = vel_r
            
            if z_ind == half_pts:
                vel_rad_ln[d, y_ind] = vel_r

vel_min = np.min( vel_rad )
vel_max = np.max( vel_rad )
vel_hlf = np.linspace(vel_min, 3*vel_max, pts)[251]
    
fnt = 18
fig, ax1 = plt.subplots()

if svar.fluid == 'Plasma':
    color = 'r'
elif svar.fluid == 'Neutral':
    color = 'b'
    
t_mark =  round( svar.t_scl * svar.dt * svar.save_steps[svar.dump])
ax1.text(0.215, 0.9, 'time: {} ns'.format(t_mark), bbox={'facecolor':'white', 'alpha':1, 'pad':5}, 
         transform=ax1.transAxes, ha='center', fontsize=fnt, animated=True)

ax1.imshow(vel_rad[svar.dump], extent=(z_dom[0], z_dom[-1], y_dom[0], y_dom[-1]), vmin=vel_min, vmax=vel_max, animated=True)
ax1.set_xlabel('Z (m)', fontsize=fnt)
ax1.set_xlim(z_dom[0], z_dom[-1])
ax1.set_ylabel('Y (mm)', fontsize=fnt)
ax1.set_ylim(y_dom[0], y_dom[-1])
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
ax2.plot(z_dom, vel_rad_ln[svar.dump], color=color, label='line out')
ax2.set_ylabel('{} Velocity (km/s)'.format(svar.fluid), color=color, fontsize=fnt)
ax2.set_ylim(vel_min, 3*vel_max)
ax2.plot([z_dom[half_pts], z_dom[half_pts]], [vel_min, 3*vel_max], color=color, ls='--',  label='line out path')
ax2.set_xlim(z_dom[0], z_dom[-1])
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.savefig(svar.vel_path+'long_frame_{}.png'.format(svar.dump))

plt.close()