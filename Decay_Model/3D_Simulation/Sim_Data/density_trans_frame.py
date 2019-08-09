# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:17:30 2019

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
  
myFile = h5.File('Data/'+svar.name+'.h5', 'r')

den = myFile[svar.fluid+'/Density/time_0']

den_z_ind = int( .5 * den.shape[2] )

x_dom_og = var.X_dom_1D * svar.s_scl * 1e3

pts = 501
half_pts = int(.5*pts)

x_dom = np.linspace(x_dom_og[0], x_dom_og[-1], pts) 

den = myFile['{0}/Density/time_{1}'.format(svar.fluid, svar.dump)][0::, 0::, den_z_ind] 
den_spline = sci.RectBivariateSpline(x_dom_og, x_dom_og, den)    

line = np.empty(pts)
im = np.empty((pts, pts))
for x_ind, x in enumerate(x_dom):
    for y_ind, y in enumerate(x_dom):
        im[x_ind, y_ind] = den_spline(x, y)[0][0]
        
        if x_ind == half_pts:
            line[y_ind] = den_spline(x_dom[half_pts], y)[0][0]

fnt = 18
fig, ax1 = plt.subplots()

if svar.fluid == 'Plasma':
    color = 'r'
elif svar.fluid == 'Neutral':
    color = 'b'
    
t_mark =  round( svar.t_scl * svar.dt * svar.save_steps[svar.dump])
text = 'time: {} ns'.format(t_mark)
texts1 = ax1.text(0.215, 0.9, text, bbox={'facecolor':'white', 'alpha':1, 'pad':5}, 
     transform=ax1.transAxes, ha='center', fontsize=fnt, animated=True)

ax1.imshow(im, extent=(x_dom[0], x_dom[-1], x_dom[0], x_dom[-1]), vmin=0, vmax=1, animated=True)
ax1.set_xlabel('X (mm)', fontsize=fnt)
ax1.set_xlim(x_dom[0], x_dom[-1])
ax1.set_ylabel('Y (mm)', fontsize=fnt)
ax1.set_ylim(x_dom[0], x_dom[-1])
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
ax2.plot(x_dom, line, color=color, label='line out')
ax2.set_ylabel('{} Density'.format(svar.fluid), color=color, fontsize=fnt)
ax2.set_xlim(x_dom[0], x_dom[-1])
ax2.set_ylim(0, 3)
ax2.plot([x_dom[0], x_dom[-1]], [1.505, 1.505], color=color, ls='--',  label='line out path')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.savefig(svar.den_path+'trans_frame_{}.png'.format(svar.dump))

plt.close()