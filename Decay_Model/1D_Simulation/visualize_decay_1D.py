#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:13:50 2019

@author: michael
"""
import global_variables as var

import h5py as h5
import numpy as np                  
from matplotlib import pyplot as plt  
from matplotlib import animation as an    

myFile = h5.File(var.data_file, 'r')
rng = len(myFile['Plasma/Density'])

sub_divide = 1
frames = int(rng / sub_divide)

frame_step = var.t_end / frames

plasma_density = np.empty((frames, var.Rpts))
neutral_density = np.empty((frames, var.Rpts))

plasma_velocity = np.empty((frames, var.Rpts))
neutral_velocity = np.empty((frames, var.Rpts))

for i in range(frames):
    key = 'time_{}'.format(i * sub_divide)
    
    plasma_density[i] = myFile['Plasma/Density/'+key][:]
    neutral_density[i] = myFile['Neutral/Density/'+key][:]
    
    plasma_velocity[i] = myFile['Plasma/Velocity/'+key][:]
    neutral_velocity[i] = myFile['Neutral/Velocity/'+key][:]
    
x = var.R * var.s * 10e3
    
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(wspace=0.45)

box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width, box2.height * 0.9])

dom = 50
ax1.set_xlim((x[0], x[-1]))
#ax1.set_xlim((-dom, dom))
ax1.set_ylim((-0.1, 1.5))
ax1.set_xlabel('(mm)')
ax1.set_ylabel('Normalized Density')

ax2.set_xlim((x[0], x[-1]))
#ax2.set_xlim((-dom, dom))
ax2.set_ylim((-4, 4))
ax2.set_xlabel('(mm)')
ax2.set_ylabel('Fluid Velocity (km/s)')

line1, = ax1.plot([], [])
line2, = ax2.plot([], [])

plotcols, plotlabs1, plotlabs2 = ["black","red"], ["Neutral Density", "Plasma Density"], ["Neutral Velocity", "Plasma Velocity"]
lines1 = []
lines2 = []
for index in range(2):
    lobj1 = ax1.plot([], [], color=plotcols[index], label=plotlabs1[index])[0]
    lobj2 = ax2.plot([], [], color=plotcols[index], label=plotlabs2[index])[0]
    lines1.append(lobj1)
    lines2.append(lobj2)
    
title1 = ax1.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax1.transAxes, ha="center")

def animate_density(i):
    
    y1 = neutral_density[i]
    y2 = plasma_density[i]

    xlist = [x, x]
    ylist = [y1, y2]
    
    for lnum,line1 in enumerate(lines1):
        line1.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
        
    y1 = var.v * neutral_velocity[i] * 10**-3
    y2 = var.v * plasma_velocity[i] * 10**-3

    xlist = [x, x]
    ylist = [y1, y2]

    for lnum,line2 in enumerate(lines2):
        line2.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
    
    time_title = round(frame_step * i * 10**9)
    title1.set_text(u'$Time: \, {} \, (ns)$'.format(time_title))
    
    return lines1, lines2, title1

ani = an.FuncAnimation(fig, animate_density, frames=frames)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
#fig.tight_layout()

writer = an.FFMpegWriter(fps=50)
ani.save(var.viz_dirc+var.shared_Name+".mp4", writer=writer)