#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:07:38 2022

@author: loewend
"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import numpy as np
from IPython.display import display

# https://github.com/endolith/bipolar-colormap.git
from bipolar import bipolar, hotcold



def image(B0, absS, bl_water, bl_fat):
    fig, ax = plt.subplots(figsize=(8, 6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
               [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
               [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619], 
               [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333], 
               [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
               [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429], 
               [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952], 
               [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
               [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286], 
               [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714], 
               [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135], 
               [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
               [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476], 
               [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143], 
               [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
               [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333], 
               [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714], 
               [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
               [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857], 
               [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857], 
               [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
               [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
               [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857], 
               [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
               [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
               [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571], 
               [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619], 
               [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
               [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
               [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
               [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952], 
               [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]
    
    slices = B0.shape[2]
    coils = absS.shape[0]
    
    # create interaction box
    slc = widgets.IntSlider(
        value=slices//2, 
        min=1, max=slices, step=1, 
        description='slice'
    )
    channel = widgets.IntSlider(
        value=1, 
        min=1, max=coils, step=1, 
        description='channel'
    )
    disp = widgets.RadioButtons(
        value='Bloch fat sim', 
        options=['B0', 'S', 'Bloch water sim', 'Bloch fat sim'], 
        description='display:'
    )
    vm = widgets.Text(
        value='11', 
        description='max FA'
    )
    ui = widgets.HBox([disp, widgets.VBox([slc, channel, vm])])
    
    def f(disp, slc, channel, vm):
        ax.cla()
        cax.cla()
        ax.set_xlim([0, B0.shape[1]])
        ax.set_ylim([0, B0.shape[0]])
        cmap = LinearSegmentedColormap.from_list('parula', cm_data)
        cmap.set_bad(color='gray')
        
        
        
        if disp == 'B0': 
            cmap = bipolar(neutral=0)
            cmap.set_bad(color='gray')
            im = ax.imshow(B0[:,:,slc-1], cmap=cmap, vmin=-np.max(np.abs(B0)), vmax=np.max(np.abs(B0)))
            

            cbar = fig.colorbar(im, cax=cax)#, orientation="horizontal")
            cbar.set_label('[Hz]')
            ax.set_title('B0-map')
            
            
        elif disp == 'S':
            im = ax.imshow(absS[channel-1,:,:,slc-1], cmap=cmap, vmin=0, vmax=np.max(absS))
            cbar = fig.colorbar(im, cax=cax)#, orientation="vertical")
            cbar.set_label('S [µT/V]')
            ax.set_title('Sensitivity-map (of chosen channel)')
            
        elif disp == 'Bloch water sim':
            try:
                vmax = float(vm)
            except:
                vmax = 11
            im = ax.imshow(bl_water[:,:,slc-1], cmap=cmap, vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, cax=cax)#, orientation="vertical")
            cbar.set_label('FA [deg]')
            ax.set_title('Bloch simulation (on-resonant)')
        
        else:
            try:
                vmax = float(vm)
            except:
                vmax = 1
            im = ax.imshow(bl_fat[:,:,slc-1], cmap=cmap, vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, cax=cax)#, orientation="vertical")
            cbar.set_label('FA [deg]')
            ax.set_title('Bloch simulation (at fat frequency)')
    
    out = widgets.interactive_output(f, {'disp': disp, 'slc': slc, 'channel': channel, 'vm': vm})

    display(ui, out)
    
    return


def pulseshape(g, g_bino, RF, RF_bino, dt):
    fig, axarr = plt.subplots(2, figsize=(8, 6))
    fig.suptitle('RF and Gradient waveforms', fontsize=16)
    
    # create interaction box
    disp = widgets.RadioButtons(
        value='binomial pulse', 
        options=['original pulse', 'binomial pulse'], 
        description='display:'
    )
    ui = widgets.VBox([disp])
    
    def f(disp):
        axarr[0].clear()
        axarr[1].clear()
        axarr[0].set_xlim([0, (g_bino.shape[0]+20)*dt*1000])
        axarr[1].set_xlim([0, (g_bino.shape[0]+20)*dt*1000])
        plt.xlabel('time [ms]')
        
        if disp == 'binomial pulse': 
            x = np.linspace(0,g_bino.shape[0]*dt*1000,g_bino.shape[0])
            axarr[0].set_ylim([0, 1.05*np.max(np.abs(RF))])
            axarr[0].set_ylabel('RF [V]')
            axarr[0].plot(np.transpose(np.tile(x,(RF_bino.shape[0],1))), np.transpose(np.abs(RF_bino)))
            axarr[0].legend(['Channel 1', 'Channel 2', 'Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8'],
                            fontsize = 'x-small')
            
            axarr[1].set_ylim([-1.2*np.max(np.abs(g_bino)), 1.2*np.max(np.abs(g_bino))])
            plt.ylabel('Gradient [mT/m]')
            axarr[1].plot(np.transpose(np.tile(x,(g_bino.shape[1],1))),g_bino)
            axarr[1].legend(['Gx', 'Gy', 'Gz'], fontsize = 'x-small')
            
            
        else:
            x = np.linspace(0,g.shape[0]*dt*1000,g.shape[0])
            axarr[0].set_ylim([0, 1.05*np.max(np.abs(RF))])
            axarr[0].set_ylabel('RF [V]')
            axarr[0].plot(np.transpose(np.tile(x,(RF.shape[0],1))),np.transpose(np.abs(RF)))
            axarr[0].legend(['Channel 1', 'Channel 2', 'Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8'],
                            fontsize = 'x-small')
            
            axarr[1].set_ylim([-1.2*np.max(np.abs(g_bino)), 1.2*np.max(np.abs(g_bino))])
            plt.ylabel('Gradient [mT/m]')
            axarr[1].plot(np.transpose(np.tile(x,(g.shape[1],1))),g)
            axarr[1].legend(['Gx', 'Gy', 'Gz'], fontsize = 'x-small')
    
            
            
    out = widgets.interactive_output(f, {'disp': disp})
    
    display(ui, out)
    return
