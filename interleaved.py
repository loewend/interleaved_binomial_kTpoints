#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:08:15 2022

@author: loewend

"""
import numpy as np
from scipy import integrate as sint


#%%
def blip(k_start,k_end,g_max,s_max,dt):
    """

    Parameters
    ----------
    k_start : np.array
        k-space starting point [kx ky kz]  ([1/m])
    k_end : np.array
        radient end point      [kx ky kz]  ([1/m]).
    g_max : float
        maximum gradient strength [mT/m]
    s_max : float
        maximum slew rate [mT/m/ms]
    dt : float
        basic sampling time of the gradient [s] (no oversampling)

    Returns
    -------
    np.array
        'g_out':        the computed gradient trajectory [mT/m]

    """

    if np.all(k_start == k_end):
        return np.array([[np.nan, np.nan, np.nan]])
    
    gamma       = 42577;               # gyromagnetic constant [Hz/mT]
    s_max = s_max*1000  #  [mT/m/ms] -> [mT/m/s]

    k_diff = (k_end-k_start)
    k_diffm = np.max(np.abs(k_diff))
    
    
    if k_diffm <= gamma*g_max*g_max/s_max: # when gmax isnt reached make a triangluar blip
        samples = np.ceil(np.sqrt(k_diffm/s_max/gamma)/dt).astype(np.int)
        T = samples*dt
        g_out = np.concatenate((np.linspace(0, -1*k_diff[0]/T/gamma, samples+1), 
                                np.linspace(0, -1*k_diff[0]/T/gamma, samples, False)[::-1],
                                np.linspace(0, -1*k_diff[1]/T/gamma, samples+1), 
                                np.linspace(0, -1*k_diff[1]/T/gamma, samples, False)[::-1],
                                np.linspace(0, -1*k_diff[2]/T/gamma, samples+1),
                                np.linspace(0, -1*k_diff[2]/T/gamma, samples, False)[::-1]))
        g_out = np.reshape(g_out,(samples*2+1,3), order='F')

    else: # when gmax is reached make a trapezoidal blip
        T1 = np.ceil(g_max/s_max/dt).astype(np.int) # ramp time divided by dt (only up or down)
        T2 = np.ceil((k_diffm/(gamma*g_max)-T1)).astype(np.int) # flat top time divided by dt
        G = -1*k_diff/(T1+T2)/dt/gamma;
        g_out = np.concatenate((np.linspace(0, G[0], T1+1), 
                                np.ones(T2)*G[0],
                                np.linspace(0, G[0], T1, False)[::-1],
                                np.linspace(0, G[1], T1+1), 
                                np.ones(T2)*G[1],
                                np.linspace(0, G[1], T1, False)[::-1],
                                np.linspace(0, G[2], T1+1), 
                                np.ones(T2)*G[2],
                                np.linspace(0, G[2], T1, False)[::-1])) 
        g_out = np.reshape(g_out,(T1*2+T2+1,3), order='F')
    #g_out = -1*g_out # because oversample_grad has no minus sign in it

    return g_out



#%%
def pulse_shape(RF):
    """

    Parameters
    ----------
    RF : np.array
        RF-pulse information.

    Returns
    -------
    rf_detail : np.array
        RF information.
    num_kT : int
        number of kT-points in RF-pulse.

    """
    rf_detail = []
    num_kT = 0     # number of kT points
    state = 0      # is it a gradient or a rf-pulse (0/1)
    num = 0        # number of samples for rf_detail
    num_sec = 0    # number of sections in rf_detail
    
    
    
    for count_sample in np.arange(RF.shape[1]):
        if np.max(np.abs(RF[:,count_sample])) == 0:
            if state == 0:
                num += 1
            else:
                rf_detail = np.concatenate((rf_detail, [num, 1]))
                num_kT += 1
                num_sec += 1
                num = 1
                state = 0
        else:
            if state == 1:
                num += 1
            else:
                rf_detail = np.concatenate((rf_detail, [num, 0]))
                num_sec += 1
                num = 1
                state = 1          
    
    rf_detail = np.concatenate((rf_detail, [num, state]))
    num_sec += 1
    if state == 1:
        num_kT += 1
        
    return np.reshape(rf_detail,(num_sec,2)), num_kT



#%%
def calc_bino(g, RF, dt, fat_freq, g_max, s_max, coils, binOrder = 1):
    """

    Parameters
    ----------
    g : np.array
        gradient trajectory.
    RF : np.array
        RF-pulse information..
    dt : float
        scanner sampling time.
    fat_freq : float
        frequency shift of fat.
    g_max : float
        maximum gradient strength in mT/m.
    s_max : float
        maximum slew rate in mT/m/ms.
    coils : int
        number of independent channels.
    binOrder : int, optional
        binomial Order. 1 = binomial 1-1 waterexcitation. 2 = binomial 1-2-1 water excitation. The default is 1.

    Returns
    -------
    g_bino : np.array
        new binomial gradient trajectory.
    RF_bino : np.array
        new binomial RF-pulse information.

    """
    
    g_inv = g[::-1]                         # reverse trajectory for easier calculation and better quality
    rf_detail, num_kT = pulse_shape(RF)
    rf_detail_inv = rf_detail[::-1]
    num_sec = rf_detail_inv.shape[0]
    Tfat_half = 1/2/np.abs(fat_freq)

    gamma = 42577 # Hz/mT
    
    k_inv = -1*sint.cumtrapz(g_inv,dx=1, axis=0, initial=0)*gamma*dt # in 1/m
    
    interleave = np.zeros(num_kT).astype(np.int)           # at which subpulses the repetition for bino is starting
    blocks = np.int(1)                                     # in how many blocks the kTpulse is split (called parts in describtion)
    kT = np.int(0)                                         # keeps track at which kT point the loop will be
    samples = np.int(0)                                    # keeps track how many samples of the pulse have passed already in the loop
    time = np.int(0)                                       # keeps track of the time passed since last block has started. When this reaches Tfat_half a new block is started
    subpulse_timings = np.zeros((num_kT,2)).astype(np.int) # first column: start of subpulse. second column: end of subpulse
    gxy_add = []                                           # gradient blips to get from k-space position of the last kTpoint in the block to the first point in this block
    zero = np.zeros(num_kT).astype(np.int)                 # number of zeros that have to be added after a block until its repetition matches the fat cycle
    
    g_old = g_inv                # store original gradient trajectory 
    g_inv = np.zeros((1,3)) 
    if rf_detail_inv[0,1] == 0:
        g_inv = np.concatenate((g_inv, g_old[:int(rf_detail_inv[0,0])]))
        time -= rf_detail_inv[0,0]
    
    
    #%% make binomial 1-2-1 pulse
    
    if binOrder == 1:     
        for count in np.arange(num_sec):
            samples += rf_detail_inv[count,0]  # number of samples of each section is added to the total number of samples so far
            time += rf_detail_inv[count,0]     # number of samples of each section is added to the number of samples since the last block started

            if rf_detail_inv[count,1] == 1:  # if a rf-pulse is applied in this section
                subpulse_timings[kT,0] = samples-rf_detail_inv[count,0]      # starting sample of the rf-pulse
                subpulse_timings[kT,1] = samples-1                       # ending sample of the rf-pulse
        
                if time*dt > Tfat_half:           # condition to split the pulse before this rf-pulse is applied 
                    if kT <= 1:
                        print('WARNING: Subpulses are to long. Try RECT pulses instead!!!')
                    
                    interleave[blocks] = kT   # this pulse (number kT -1) will be the first rf-pulse of the next block
        
                    # add blip from last point of the block to the first point ("blocks" is the number of the next block that will start)
                    gxy_add.append(blip(k_inv[subpulse_timings[interleave[blocks]-1,1],:],
                                        k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                        g_max,s_max,dt))
    
                    # if the block together with the blip would take more than Tfat_half
                    if (subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0])*dt > Tfat_half:
                        if kT <= 2:
                            print('WARNING: Subpulses are to long. Try RECT pulses instead!!!')
                            
                        interleave[blocks] = kT-1 # define new block to start at the previous sub-pulse
        
                        # add blip from last point (new definition) of the block to the first point ("blocks" is the number of the next block that will start)
                        gxy_add[blocks-1] = blip(k_inv[subpulse_timings[kT-2,1],:],
                                                 k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                                 g_max,s_max,dt)
                        
                    zero[blocks-1]=np.round_(Tfat_half/dt-(subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0]))
                    g_inv = np.concatenate((g_inv, 
                                            g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                            gxy_add[blocks-1],
                                            np.zeros((zero[blocks-1],3)),    # add zeros until fat has 180° phase shift compared to water
                                            g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks],0],:]))  
                    
                    time = samples-subpulse_timings[interleave[blocks],0]; # start counting samples for the next block
                    blocks += 1                   # the pulse gets seperated in another block 
                
                kT += 1                  # count number of kTpoints applied so far
                
            
        # add the last block (the condition timing*Ts > Tfat_half could not be achieved)
        gxy_add.append(blip(k_inv[subpulse_timings[-1,1],:],
                            k_inv[subpulse_timings[interleave[blocks-1],0],:],
                            g_max,s_max,dt))         # get to the first kTpoint of last pulse again
        zero[blocks-1] = np.round_(Tfat_half/dt-(subpulse_timings[-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0]))
        
        
        if zero[blocks-1] < 0:
            interleave[blocks]=kT-1
            gxy_add[blocks-1] = blip(k_inv[subpulse_timings[interleave[blocks]-1,1],:],
                                     k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                     g_max,s_max,dt)
            zero[blocks-1] = np.round_(Tfat_half/dt-(subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-2].shape[0]))
            gxy_add.append(blip(k_inv[subpulse_timings[-1,1],:],
                                k_inv[subpulse_timings[interleave[blocks],0],:],
                                g_max,s_max,dt)) # get to the first kTpoint of last pulse again
            zero[blocks] = np.round_(Tfat_half/dt-(subpulse_timings[-1,1]-subpulse_timings[interleave[blocks],0]+1+gxy_add[blocks-1].shape[0]))
            g_inv = np.concatenate((g_inv[1:],
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),  # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1,:],
                                    np.zeros((zero[blocks],3)),
                                    g_old[subpulse_timings[interleave[blocks],0]:,:]))
            blocks += 1
        
        else:
            g_inv = np.concatenate((g_inv[1:],
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),   # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:,:]))
        

        # make binomial RF
        
        RF_inv = RF[:,::-1]/2   # half the amplitude, because pulse is played twice
        RF_old = RF_inv           # store information of the precalculated pulse 
        RF_inv = np.zeros((coils,1)) 
        
        if rf_detail_inv[0,1] == 0:
            RF_inv = np.concatenate((RF_inv, RF_old[:,:int(rf_detail_inv[0,0])]),axis=1)
        
        
        for count_block in np.arange(blocks-1):
            RF_inv = np.concatenate((RF_inv,
                                     RF_old[:,subpulse_timings[interleave[count_block],0]:subpulse_timings[interleave[count_block+1]-1,1]+1],
                                     np.zeros((coils,gxy_add[count_block].shape[0])),
                                     np.zeros((coils,zero[count_block])),    # add zeros until fat has 180° phase shift compared to water
                                     RF_old[:,subpulse_timings[interleave[count_block],0]:subpulse_timings[interleave[count_block+1],0]]),axis=1)
            
        
        RF_inv = np.concatenate((RF_inv[:,1:],
                                 RF_old[:,subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1],
                                 np.zeros((coils,gxy_add[blocks-1].shape[0])),
                                 np.zeros((coils,zero[blocks-1])),    # add zeros until fat has 180° phase shift compared to water
                                 RF_old[:,subpulse_timings[interleave[blocks-1],0]:]),axis=1)
        

    
    
    #%% make binomial 1-2-1 pulse
    
    elif binOrder == 2:
        for count in np.arange(num_sec):
            samples += rf_detail_inv[count,0]  # number of samples of each section is added to the total number of samples so far
            time += rf_detail_inv[count,0]     # number of samples of each section is added to the number of samples since the last block started
        
            if rf_detail_inv[count,1] == 1:  # if a rf-pulse is applied in this section
                subpulse_timings[kT,0] = samples-rf_detail_inv[count,0]      # starting sample of the rf-pulse
                subpulse_timings[kT,1] = samples-1                       # ending sample of the rf-pulse
        
                if time*dt > Tfat_half:           # condition to split the pulse before this rf-pulse is applied 
                    if kT <= 1:
                        print('WARNING: Subpulses are to long. Try RECT pulses instead!!!')
                    
                    interleave[blocks] = kT   # this pulse (number kT -1) will be the first rf-pulse of the next block
        
                    # add blip from last point of the block to the first point ("blocks" is the number of the next block that will start)
                    gxy_add.append(blip(k_inv[subpulse_timings[interleave[blocks]-1,1],:],
                                        k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                        g_max,s_max,dt))
    
                    # if the block together with the blip would take more than Tfat_half
                    if (subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0])*dt > Tfat_half:
                        if kT <= 2:
                            print('WARNING: Subpulses are to long. Try RECT pulses instead!!!')
                            
                        interleave[blocks] = kT-1 # define new block to start at the previous sub-pulse
        
                        # add blip from last point (new definition) of the block to the first point ("blocks" is the number of the next block that will start)
                        gxy_add[blocks-1] = blip(k_inv[subpulse_timings[kT-2,1],:],
                                                 k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                                 g_max,s_max,dt)
                        
                    zero[blocks-1]=np.round_(Tfat_half/dt-(subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0]))
                    g_inv = np.concatenate((g_inv, 
                                            g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                            gxy_add[blocks-1],
                                            np.zeros((zero[blocks-1],3)),    # add zeros until fat has 180° phase shift compared to water
                                            g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                            gxy_add[blocks-1],
                                            np.zeros((zero[blocks-1],3)),    # add zeros until fat has 180° phase shift compared to water
                                            g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks],0],:]))  
                    
                    time = samples-subpulse_timings[interleave[blocks],0]; # start counting samples for the next block
                    blocks += 1                   # the pulse gets seperated in another block 
                
                kT += 1                  # count number of kTpoints applied so far
                
            
            
        # add the last block (the condition timing*Ts > Tfat_half could not be achieved)
        gxy_add.append(blip(k_inv[subpulse_timings[-1,1],:],
                            k_inv[subpulse_timings[interleave[blocks-1],0],:],
                            g_max,s_max,dt))         # get to the first kTpoint of last pulse again
        zero[blocks-1] = np.round_(Tfat_half/dt-(subpulse_timings[-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0]))
        
        
        if zero[blocks-1] < 0:
            interleave[blocks]=kT-1
            gxy_add[blocks-1] = blip(k_inv[subpulse_timings[interleave[blocks]-1,1],:],
                                     k_inv[subpulse_timings[interleave[blocks-1],0],:],
                                     g_max,s_max,dt)
            zero[blocks-1] = np.round_(Tfat_half/dt-(subpulse_timings[interleave[blocks]-1,1]-subpulse_timings[interleave[blocks-1],0]+1+gxy_add[blocks-1].shape[0]))
            gxy_add.append(blip(k_inv[subpulse_timings[-1,1],:],
                                k_inv[subpulse_timings[interleave[blocks],0],:],
                                g_max,s_max,dt)) # get to the first kTpoint of last pulse again
            zero[blocks] = np.round_(Tfat_half/dt-(subpulse_timings[-1,1]-subpulse_timings[interleave[blocks],0]+1+gxy_add[blocks].shape[0]))
            g_inv = np.concatenate((g_inv[1:],
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),  # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[interleave[blocks]-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),  # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1,:],
                                    gxy_add[blocks],
                                    np.zeros((zero[blocks],3)),
                                    g_old[subpulse_timings[interleave[blocks],0]:subpulse_timings[-1,1]+1,:],
                                    gxy_add[blocks],
                                    np.zeros((zero[blocks],3)),
                                    g_old[subpulse_timings[interleave[blocks],0]:,:]))
            blocks += 1
        
        else:
            g_inv = np.concatenate((g_inv[1:],
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),   # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1,:],
                                    gxy_add[blocks-1],
                                    np.zeros((zero[blocks-1],3)),   # add zeros until fat has 180° phase shift compared to water
                                    g_old[subpulse_timings[interleave[blocks-1],0]:,:]))
        
    
        # make binomial RF
        
        RF_inv = RF[:,::-1]/4   # half the amplitude, because pulse is played twice
        RF_old=RF_inv           # store information of the precalculated pulse        
        RF_inv = np.zeros((coils,1))
        
        if rf_detail_inv[0,1] == 0:
            RF_inv = np.concatenate((RF_inv, RF_old[:,:int(rf_detail_inv[0,0])]),axis=1)
        
        for count_block in np.arange(blocks-1):
            RF_inv = np.concatenate((RF_inv,
                                     RF_old[:,subpulse_timings[interleave[count_block],0]:subpulse_timings[interleave[count_block+1]-1,1]+1],
                                     np.zeros((coils,gxy_add[count_block].shape[0])),
                                     np.zeros((coils,zero[count_block])),    # add zeros until fat has 180° phase shift compared to water
                                     RF_old[:,subpulse_timings[interleave[count_block],0]:subpulse_timings[interleave[count_block+1]-1,1]+1]*2,
                                     np.zeros((coils,gxy_add[count_block].shape[0])),
                                     np.zeros((coils,zero[count_block])),    # add zeros until fat has 180° phase shift compared to water
                                     RF_old[:,subpulse_timings[interleave[count_block],0]:subpulse_timings[interleave[count_block+1],0]]),axis=1)
            
        RF_inv = np.concatenate((RF_inv[:,1:],
                                 RF_old[:,subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1],
                                 np.zeros((coils,gxy_add[blocks-1].shape[0])),
                                 np.zeros((coils,zero[blocks-1])),    # add zeros until fat has 180° phase shift compared to water
                                 RF_old[:,subpulse_timings[interleave[blocks-1],0]:subpulse_timings[-1,1]+1]*2,
                                 np.zeros((coils,gxy_add[blocks-1].shape[0])),
                                 np.zeros((coils,zero[blocks-1])),    # add zeros until fat has 180° phase shift compared to water
                                 RF_old[:,subpulse_timings[interleave[blocks-1],0]:]),axis=1)
    
    RF_inv = RF_inv[:,~np.isnan(g_inv)[:,0]]
    g_inv = g_inv[~np.isnan(g_inv)[:,0],:]

        
    
    return g_inv[::-1], RF_inv[:,::-1]
    
