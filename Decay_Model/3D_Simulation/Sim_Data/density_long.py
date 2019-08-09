# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:58:01 2019

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
from matplotlib import animation as an  
  
myFile = h5.File('Data/'+svar.name+'.h5', 'r')
den = myFile['{0}/Density/time_0'.format(svar.fluid)]

den_x_ind = int( .5 * den.shape[0] )
den_z_ind = int( .5 * den.shape[2] )

den_long = den[den_x_ind, 0::, 0::] 

x_dom_og = var.X_dom_1D * svar.s_scl * 1e3
z_dom_og = (var.Z_dom_1D - var.Z_dom_1D[0]) * svar.s_scl

pts = 501
half_pts = int(.5*pts)

x_dom = np.linspace(x_dom_og[0], x_dom_og[-1], pts) 
z_dom = np.linspace(z_dom_og[0], z_dom_og[-1], pts) 
    
fnt = 18
fig, ax1 = plt.subplots()

if svar.fluid == 'Plasma':
    color = 'r'
elif svar.fluid == 'Neutral':
    color = 'b'

ax1.set_xlabel('Z (m)', fontsize=fnt)
ax1.set_xlim(z_dom[0], z_dom[-1])
ax1.set_ylabel('Y (mm)', fontsize=fnt)
ax1.set_ylim(x_dom[0], x_dom[-1])
ax1.tick_params(axis='y', labelcolor='k')

ax2 = ax1.twinx()
ax2.set_ylabel('{} Density'.format(svar.fluid), color=color, fontsize=fnt)
ax2.set_xlim(z_dom[0], z_dom[-1])
ax2.set_ylim(0, 3)
ax2.plot([z_dom[0], z_dom[-1]], [1.505, 1.505], color=color, ls='--',  label='line out path')
ax2.tick_params(axis='y', labelcolor=color)

artist = []
for d in range(svar.dumps):
    t_mark =  round( svar.t_scl * svar.dt * svar.save_steps[d])
    text = 'time: {} ns'.format(t_mark)
    texts1 = ax1.text(0.215, 0.9, text, bbox={'facecolor':'white', 'alpha':1, 'pad':5}, 
         transform=ax1.transAxes, ha='center', fontsize=fnt, animated=True)
    
    den = myFile['{0}/Density/time_{1}'.format(svar.fluid, d)]
    den_long = den[den_x_ind, 0::, 0::] 

    den_long_spline = sci.RectBivariateSpline(x_dom_og, z_dom_og, den_long)
    
    line = np.empty(pts)
    im = np.empty((pts, pts))
    for x_ind, x in enumerate(x_dom):
        for z_ind, z in enumerate(z_dom):
            im[x_ind, z_ind] = den_long_spline(x, z)[0][0]
            
            if x_ind == half_pts:
                line[z_ind] = den_long_spline(x_dom[half_pts], z)[0][0]
    
    image = ax1.imshow(im, extent=(z_dom[0], z_dom[-1], x_dom[0], x_dom[-1]), vmin=0, vmax=1, animated=True)
    image1 = image
    
    if d == 0:
        line1, = ax2.plot(z_dom, line, color=color, label='line out')
    else:
        line1, = ax2.plot(z_dom, line, color=color)
    
    artist.append([image1, texts1, line1])
    
ax2.legend(loc='upper right')

#cbaxes = fig.add_axes([0.125, .9, 0.775, 0.03])
#plt.colorbar(image, cax=cbaxes, orientation='horizontal')

ani = an.ArtistAnimation(fig, artist)
writer = an.FFMpegWriter(fps=10, codec='h264')
ani.save(svar.den_path+'movies/long.mp4', writer=writer)

plt.close()