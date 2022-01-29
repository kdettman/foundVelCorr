import math
import numpy as np
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import sys
import os
import copy
from scipy.interpolate import interp1d
import pandas as pd
from scipy.signal import savgol_filter


p_init = models.Polynomial1D(degree=2)    
g_init = models.Gaussian1D(amplitude=3.0e-16, mean=0, stddev=0.3)
g2_init = models.Gaussian1D(amplitude=1.0, mean=-10000, stddev=100)
gp_init = p_init - g_init
gp_init2 = p_init - g2_init

si2wav = 6355.0 #angstroms
c = 299792.458 #km/s


def resClip(data_x, data_y, model, sigma):
    '''
    Takes a set of data and a model based on that data and returns a new set
    of data whose residuals have been sigma clipped
    '''
    removed = False
    residuals = data_y - model(data_x)
    stddev = np.std(residuals)
    index = []
    mask_x = np.ones(len(data_x), np.bool)
    mask_y = np.ones(len(data_y), np.bool)
    for i in range(len(data_x)):
        if abs(residuals[i]) > sigma*stddev:
            try:
                #del data_x_cp[i]
                #del data_y_cp[i]
                #keeping track of what indices have bad points
                index.append(i)
                removed = True
            except IndexError:
                break
    
    mask_x[index]=False
    mask_y[index]=False
    data_y_cp = np.asarray(data_y)[mask_y]
    data_x_cp = np.asarray(data_x)[mask_x]
    data_x_ret = data_x_cp.tolist()
    data_y_ret = data_y_cp.tolist()
    return data_x_ret, data_y_ret, removed

def gausfit(fileName, center, width, z, zerr, name, clip=False, sigma=3, center_guess=None, fwhm=None, interpolate=False, plot=False, polydeg=2, ex_range=1000):
    '''
    Fits a gaussian + polynomial profile to a spectrum for values between center - width
    and center + width
    '''
    #Defining globals
    p_init = models.Polynomial1D(degree=polydeg)
    g_init = models.Gaussian1D(amplitude=3.0e-16, mean=0, stddev=0.3)
    g2_init = models.Gaussian1D(amplitude=1.0, mean=-10000, stddev=100)
    gp_init = p_init - g_init
    gp_init2 = p_init - g2_init
    si2wav = 6355.0 #angstroms
    c = 299792.458 #km/s

    #Importing file
    center = float(center)
    width = float(width)
    file = pd.read_csv(fileName, delim_whitespace=True, comment='#', names=['wavelength','flux'])
    wavelength = file['wavelength'].values
    flux = file['flux'].values
    
    #Correcting data for redshift
    wavelength = [x/(1.0+z) for x in wavelength]
    center = center/(1.0+z)
    if (center_guess is not None):
    	center_guess = center_guess/(1.0+z)
    if (fwhm is not None):
    	fwhm=fwhm/(1.0+z)
    
    #finding endpoints to be used in the fit
    endpointInd = min(range(len(wavelength)),key=lambda i: abs(wavelength[i]-center-width))
    farpointInd = min(range(len(wavelength)),key=lambda i: abs(wavelength[i]-center+width))

    wavefit = wavelength[farpointInd:endpointInd]
    wavefittmp = wavelength[farpointInd:endpointInd]
    fluxfit = flux[farpointInd:endpointInd]
    fluxfittmp = flux[farpointInd:endpointInd]
    endpoint = wavelength[endpointInd]
    farpoint = wavelength[farpointInd]
    scale = abs(center-width)

    
    #Changing wavelength to velocity
    velfit = [((x**2 - si2wav**2)/(x**2 + si2wav**2))*c for x in wavefit]
    velCenter = ((center**2 - si2wav**2)/(center**2 + si2wav**2))*c
    #changing flux to velocity-corrected units
    velflux = []
    for i in range(len(wavefit)):
        corr = (c*si2wav*(c-velfit[i])**(-3./2.))/np.sqrt(c+velfit[i])
        velflux.append(corr*fluxfit[i])
    
    velfit_full = [((x**2 - si2wav**2)/(x**2 + si2wav**2))*c for x in wavelength]
    velflux_full = []
    for i in range(len(wavelength)):
        corr = (c*si2wav*(c-velfit_full[i])**(-3./2.))/np.sqrt(c+velfit_full[i])
        velflux_full.append(corr*flux[i])
    
    #Correcting center_guess and fwhm to be in velocity space
    if (fwhm is not None):
    	fwhm = ((fwhm**2 - si2wav**2)/(fwhm**2 + si2wav**2))*c
    if (center_guess is not None):
    	center_guess = ((center_guess**2 - si2wav**2)/(center_guess**2 + si2wav**2))*c
    	#print 'Center guess in vel: ' + str(center_guess)
    	
    #interpolating additional points into the data
    if (interpolate):
    	interp = interp1d(velfit, velflux, kind='cubic')
    	velfit = np.linspace(velfit[0],velfit[-1],num=2*len(velfit)).tolist()
    	velflux = interp(velfit).tolist()
    	
    #Correcting data so that the wavelength bounded by [-1,1] 
    #or is at least [-1,1) or (-1,1]
    wavefit = [(x-center) for x in wavefit]
    scale = abs(max(abs(wavefit[0]), abs(wavefit[-1])))
    wavefit = [x/scale for x in wavefit]
    #Correcting velocity data for same bounds
    velflux = [(x-min(velflux)) for x in velflux]
    velflux_full = [(x-min(velflux)) for x in velflux_full]
    
    scaleflux = max(velflux)
    velflux = [x/scaleflux for x in velflux]
    velflux_full = [x/scaleflux for x in velflux_full]
    
    idx_left = (np.abs(np.array(velfit_full) - velfit[0])).argmin()
    idx_right = (np.abs(np.array(velfit_full) - velfit[-1])).argmin()
    offset = min(velflux_full[idx_left:idx_right]) - min(velflux)
    velflux_full = [x-offset for x in velflux_full]
    
    fitter = fitting.LevMarLSQFitter()
    #Creating a new gaussian fit to include the fwhm guess
    if (center_guess is not None or fwhm is not None):
    	if (fwhm is not None):
    		std_dev = fwhm/2.355
    		g2_init = models.Gaussian1D(amplitude=1.0, mean=-10000, stddev=std_dev)
    	elif (center_guess is not None):
    		g2_init = models.Gaussian1D(amplitude=1.0, mean=center_guess, stddev=100)
    	else:
    		std_dev = fwhm/2.355
    		g2_init = models.Gaussian1D(amplitude=1.0, mean=center_guess, stddev=std_dev)
    	gp_init2 = p_init - g2_init
    fitflux = fitter(gp_init2, velfit, velflux, maxiter=1000)
    
    try:
        err1 = np.sqrt(float(fitter.fit_info['param_cov'][4][4]))
    except:
        err1 = -1
    
    vel = float(fitflux.mean_1.value)
    stdvel = np.std(velflux-fitflux(velfit))
    
    N = 31
    velCheck = []
    residuals_smooth = velflux - fitflux(velfit)
    residuals_2_smooth = residuals_smooth**2.0
    residuals = np.sqrt(residuals_2_smooth + np.std(residuals_smooth)**2.0)
    for i in range(N):
        gausflux = velflux + np.random.normal(scale=residuals)
        fitCheck = fitter(gp_init2, velfit, gausflux)
        velCheck.append(float(fitCheck.mean_1.value))
        
    stdCheck = np.std(velCheck)
    clippedStdCheck = np.sqrt(2.0)*np.std(sigma_clip(velCheck))
    
    ##### Clipped Data #####
    if (clip):
        removed = True
        velfit_clip = velfit
        velflux_clip = velflux
        fitflux_clip = fitter(gp_init2, velfit_clip, velflux_clip, maxiter=1000)

    velCheck_clip = []
    while(removed): #Iterating clipping until no more points are removed
		velfit_clip, velflux_clip, removed = resClip(velfit_clip, velflux_clip, fitflux_clip, sigma)
		fitflux_clip = fitter(gp_init2, velfit_clip, velflux_clip, maxiter=1000)

		try:
				err1_clip = np.sqrt(float(fitter.fit_info['param_cov'][4][4]))
		except:
				err1_clip = -1
					
		vel_clip = float(fitflux_clip.mean_1.value)
		stdvel_clip = np.std(velflux_clip-fitflux_clip(velfit_clip))

		residuals_smooth_clip = velflux_clip - fitflux_clip(velfit_clip)
		residuals_2_smooth_clip = residuals_smooth_clip**2.0
		residuals_clip = np.sqrt(residuals_2_smooth_clip + np.std(residuals_smooth_clip)**2.0)

    for i in range(N):
    	gausflux_clip = velflux_clip + np.random.normal(scale=residuals_clip)
    	fitCheck_clip = fitter(gp_init2, velfit_clip, gausflux_clip)
    	velCheck_clip.append(float(fitCheck_clip.mean_1.value))

    stdCheck_clip = np.std(velCheck_clip)
    clippedStdCheck_clip = np.sqrt(2.0)*np.std(sigma_clip(velCheck_clip))
    
    
    if (plot):
		# plt.figure(figsize=(8,5))
# 		plt.plot(velfit_full, velflux_full, 'y')
# 		plt.plot(velfit, velflux, 'g.')
# 		plt.plot(velfit, fitflux(velfit),'r-')
# 		#plt.xlabel('Velocity')
# 		#plt.ylabel('Flux (Scaled)')
# 		plt.xlim(min(velfit)-ex_range,max(velfit)+ex_range)
# 		plt.xticks([])
# 		plt.yticks([])
# 		plt.title(str(name))
# 		plt.savefig('VelFits/'+str(name)+'.png')
   
		if (clip):
			plt.figure(figsize=(8,5))
			plt.plot(velfit_full, velflux_full, 'y')
			plt.plot(velfit, velflux, 'b.')
			plt.plot(velfit_clip, velflux_clip, 'g.')
			plt.plot(velfit_clip, fitflux_clip(velfit_clip),'r-')
			# plt.xlabel('Velocity')
			#plt.ylabel('Flux (Scaled)')
			plt.xlim(min(velfit)-ex_range,max(velfit)+ex_range)
			plt.xticks([])
			plt.yticks([])
			plt.title(str(name)+'_clipped')
# 			plt.savefig('VelFits/'+str(name)+'_clipped.png')
    
            
    if (clip):
        return vel, err1, clippedStdCheck, vel_clip, err1_clip, clippedStdCheck_clip
    else:
        return vel, err1, clippedStdCheck, 0, 0, 0