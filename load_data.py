#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:02:03 2022

@author: loewend
"""

import scipy.io as sio
import numpy as np


#%%
def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)



#%%
def pulse_mat(file):
    mat_struct = sio.loadmat(file)
    dt = mat_struct['g_step'].astype(np.single)[0][0] # in ms
    RF = mat_struct['b_est'].astype(np.csingle) # in V   
    g = mat_struct['g_low'].astype(np.single) # in mT/m
    
    return g, RF, dt



#%%
def read_ini(file):
    f = open(file, 'r')
    fid = f.readlines()
    
    comment = '#'
    
    l = 0
    for line in fid:
        l += 1
        line = line.strip() # remove withespaces at beginning and end
        
        # remove comments
        sind = line.find(comment)
        if sind >= 0:
            curline = line[:sind].strip()
        else:
            curline = line
        
        # determine section
        sindBegin = curline.find('[')
        sindEnd = curline.find(']')
        if (sindBegin>=0 and sindEnd >=0 and sindEnd-sindBegin > 1):
            stmp = curline[sindBegin+1:sindEnd]
            if sindBegin > 0:
                key = curline[:sindBegin]
                try:
                    arrayInd = int(np.floor(float(stmp)))
                except:
                    print('Invalid format')
                    continue
            else:
                curSec = stmp
        
        # ignore empty lines
        if not curline:
            continue
        
        # read vlaues
        sindToken = curline.find('=')
        if sindToken > 0:
            token = curline[:sindToken].strip()
            remain = curline[sindToken+1:].strip()
            
            if curSec == 'pTXPulse':
                if token == 'NUsedChannels':
                    coils = int(remain)
                elif token == 'DimGradient':
                    dim_grad = int(remain)
                elif token == 'NominalFlipAngle':
                    FA = float(remain)
                elif token == 'Samples':
                    samples = int(remain)
                    RF = np.zeros((coils,samples)).astype(np.cdouble)
                else:
                    continue
            
            elif curSec == 'Gradient':
                if token == 'GradRasterTime':
                    dt = float(remain)*1e-6 # in s
                elif token == 'GradientSamples':
                    g = np.zeros((int(remain),dim_grad))
                elif key == 'G':
                    val = remain.split()
                    for count in np.arange(dim_grad):
                        g[arrayInd,count] = float(val[count])
                else:
                    continue
                        
            elif curSec[:11] == 'pTXPulse_ch':
                numCoil = int(curSec[11])
                if key == 'RF':
                    val = remain.split()
                    RF[numCoil,arrayInd] = float(val[0])*np.exp(1j*float(val[1]))
                else:
                    continue
    
    
    f.close()
    return g, RF, dt



#%%
def write_ini(infile, binOrder, g, RF):
    """
    Creates a new ini-file with parameters from infile but new gradients and RF-pulses
    
    Parameters
    ----------
    infile : str
        copy parameters from this original pulse.
    binOrder : int (1 or 2)
        create new file with appendix dependend on the order of the binomial pulse
    g : np.array
        new gradient trajectory.
    RF : np.array
        new RF-pulse information.


    """
    
    outfile = infile[:-4]+'_b'+str(binOrder)+'.ini'
    out = open(outfile, 'w')
    
    f = open(infile, 'r')
    fid = f.readlines()
    
    comment = '#'
    
    l = 0
    for line in fid:
        l += 1
        line = line.strip() # remove withespaces at beginning and end
        
        if l == 2:
            if binOrder == 1:
                bin_str = '1-1'
                out.write('#pTXRFPulse - binomial '+bin_str +' water excitation - adjusted by: <write_ini>'+'\n')
            elif binOrder == 2:
                bin_str = '1-2-1'
                out.write('#pTXRFPulse - binomial '+bin_str +' water excitation - adjusted by: <write_ini>'+'\n')
            else:
                out.write('#pTXRFPulse - adjusted by: <write_ini>'+'\n')
            
        
        # remove comments
        sind = line.find(comment)
        if sind >= 0:
            curline = line[:sind].strip()
        else:
            curline = line
        
        # determine section
        sindBegin = curline.find('[')
        sindEnd = curline.find(']')
        if (sindBegin>=0 and sindEnd >=0 and sindEnd-sindBegin > 1):
            stmp = curline[sindBegin+1:sindEnd]
            if sindBegin > 0:
                key = curline[:sindBegin]
                try:
                    arrayInd = int(np.floor(float(stmp)))
                except:
                    print('Invalid format')
                    out.write(line+'\n')
                    continue
            else:
                curSec = stmp
        
        # ignore empty lines
        if not curline:
            out.write(line+'\n')
            continue
        
        # read vlaues
        sindToken = curline.find('=')
        if sindToken > 0:
            token = curline[:sindToken].strip()
            remain = curline[sindToken+1:].strip()
            
            if curSec == 'pTXPulse':
                if token == 'DimGradient':
                    dim_grad = int(remain)
                elif token == 'MaxAbsRF':
                    out.write(token+'         = '+str(np.max(np.abs(RF)))+'\t\t # scaling for RF amplitude\n')
                    continue
                elif token == 'Samples':
                    samples = int(g.shape[0])
                    out.write(token+'          = '+str(samples)+'\n')
                    continue
                else:
                    out.write(line+'\n')
                    continue
            
            elif curSec == 'Gradient':
                if token == 'GradientSamples':
                    samples = int(g.shape[0])
                    out.write(token+'   =  '+str(samples)+'\n')
                    continue
                elif token == 'MaxAbsGradient[0]':
                    out.write(token+' =  '+str(np.max(np.abs(g[:,0])))+
                              '\t '+str(np.max(np.abs(g[:,1])))+
                              '\t '+str(np.max(np.abs(g[:,2])))+
                              '\t\t # scaling for G amplitude\n')
                    continue
                elif token == 'G[0]':
                    for count_sample in np.arange(samples):
                        out_str = 'G['+str(count_sample)+']= '
                        for count_dim in np.arange(dim_grad):
                            out_str += str(g[count_sample,count_dim])+'\t '
                        out.write(out_str+'\n')
                    continue
                elif key == 'G':
                    continue
                else:
                    out.write(line+'\n')
                    continue
                        
            elif curSec[:11] == 'pTXPulse_ch':
                numCoil = int(curSec[11])
                if token == 'RF[0]':
                    ang = np.angle(RF[numCoil])
                    ang[ang < 0] += 2*np.pi
                    for count_sample in np.arange(samples):
                        out.write('RF['+str(count_sample)+']= '+str(np.abs(RF[numCoil,count_sample]))+
                                  '\t '+str(ang[count_sample])+'\n')
                    continue
                elif key == 'RF':
                    continue
                else:
                    out.write(line+'\n')
                    continue
                
        out.write(line+'\n')
        
    out.close()
    f.close()
    return 0