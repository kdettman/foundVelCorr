import emcee
import corner
import numpy as np
import matplotlib.pylab as plt
import copy
import linmix

'''
Methods contained herin which can be used to run various fitting analyses:
split_data - segmets data based on velocity
plot_splitdata - plots a split_data object
doMCMC - runs an MCMC fitter on split_data object
dolinmix - runs linmix fitter on split_data object
dolinmix_sep - alternate method of running linmix on split_data object
'''

def split_data(data, vsplit=-11.8):

    lv = (data['vel'] >= vsplit)
    hv = (data['vel'] < vsplit)

    x_lv = (data['x'])[lv]
    y_lv = (data['y'])[lv]
    x_hv = (data['x'])[hv]
    y_hv = (data['y'])[hv]

    if ('xerr' in data):
        x_lv_err = (data['xerr'])[lv]
        x_hv_err = (data['xerr'])[hv]
    else:  
        x_lv_err = 0.0
        x_hv_err = 0.0

    if ('yerr' in data):
        y_lv_err = (data['yerr'])[lv]
        y_hv_err = (data['yerr'])[hv]
    else:
        y_lv_err = 0.0
        y_hv_err = 0.0

    v_lv = (data['vel'])[lv]
    v_hv = (data['vel'])[hv]

    spdat = { 
        'x_lv': x_lv,
        'y_lv': y_lv,
        'x_lv_err': x_lv_err,
        'y_lv_err': y_lv_err,
        'v_lv': v_lv,
        'x_hv': x_hv,
        'y_hv': y_hv,
        'x_hv_err': x_hv_err,
        'y_hv_err': y_hv_err,
        'v_hv': v_hv,
        'vsplit': vsplit,
        'n_lv': len(x_lv),
        'n_hv': len(x_hv)
        }

    return spdat


def plot_splitdata(splitdata):
    plt.errorbar(splitdata['x_lv'],splitdata['y_lv'],
                 splitdata['y_lv_err'],splitdata['x_lv_err'],'.b')
    plt.errorbar(splitdata['x_hv'],splitdata['y_hv'],
                 splitdata['y_hv_err'],splitdata['x_hv_err'],'.r')
    plt.gca().invert_yaxis()
    plt.show()


def get_fullparam(theta):

    nparam = len(theta)
    
    # 0th parameter is y at x=0 of low velocity objects
    # 1st parameter is slope (for both if nparam <= 4)
    # 2nd parameter is x offset for high velocity
    # 3rd parameter is sigma (for both if nparam <= 5)
    # 4th parameter is slope for hv (optional)
    # 5th parameter is sigma for hv (optional)

    assert (nparam >= 2) and (nparam <= 6), "invalid nparam"

    if nparam == 2:
        zeropoint, slope_lv = theta
        offset = 0.
        sigma_lv = 0.
        slope_hv = slope_lv
        sigma_hv = sigma_lv
    elif nparam == 3:
        zeropoint, slope_lv, offset = theta
        sigma_lv = 0.
        slope_hv = slope_lv
        sigma_hv = sigma_lv
    elif nparam == 4:
        zeropoint, slope_lv, offset, sigma_lv = theta
        slope_hv = slope_lv
        sigma_hv = sigma_lv
    elif nparam == 5:
        zeropoint, slope_lv, offset, sigma_lv, slope_hv = theta
        sigma_hv = sigma_lv
    elif nparam == 6:
        zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = theta

    return zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv


def log_prior(theta):
    
    nparam = len(theta)
    zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = get_fullparam(theta)
    
    # logpr = -1.5 * np.log(1.0 + slope_lv**2)     check with this turned on
    logpr = 0.

    if nparam <= 3:
        return logpr

    if (sigma_lv <= 0) or (sigma_hv <= 0):
        return -np.inf
        
    if nparam >= 4:
        logpr -= np.log(sigma_lv)

    if nparam >= 5:
        # logpr -= 1.5 * np.log(1.0 + slope_hv**2)
        logpr -= 0.

    if nparam >= 6:
        logpr -= np.log(sigma_hv)

    return logpr

def log_prior_mod(theta):
    
    nparam = len(theta)
    zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = get_fullparam(theta)
    
    logpr = -1.5 * np.log(1.0 + slope_lv**2) #    check with this turned on
    # logpr = 0.

    if nparam <= 3:
        return logpr

    if (sigma_lv <= 0) or (sigma_hv <= 0):
        return -np.inf
        
    if nparam >= 4:
        logpr -= np.log(sigma_lv)

    if nparam >= 5:
        logpr -= 1.5 * np.log(1.0 + slope_hv**2)
        # logpr -= 0.

    if nparam >= 6:
        logpr -= np.log(sigma_hv)

    return logpr


def log_likelihood(theta, splitdata):
    
    zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = get_fullparam(theta)
    
    var_lv = (splitdata['y_lv_err']**2 + 
             (slope_lv*splitdata['x_lv_err'])**2 +
             sigma_lv**2)
    model_lv = zeropoint + slope_lv * splitdata['x_lv']
    logl_lv = -0.5 * (np.sum(np.log(2 * np.pi * var_lv) + 
                            ((splitdata['y_lv'] - model_lv)**2 / var_lv) ))
 
    var_hv = (splitdata['y_hv_err']**2 + 
             (slope_hv*splitdata['x_hv_err'])**2 +
             sigma_hv**2)         
    model_hv = zeropoint + slope_hv * (splitdata['x_hv'] - offset)
    logl_hv = -0.5 * (np.sum(np.log(2 * np.pi * var_hv) + 
                            ((splitdata['y_hv'] - model_hv)**2 / var_hv) ))
        
    return logl_lv + logl_hv

    
def log_posterior(theta, splitdata):
    
    logpr = log_prior(theta)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood(theta, splitdata)

def log_posterior_mod(theta, splitdata):
    
    logpr = log_prior_mod(theta)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood(theta, splitdata)


def log_likelihood_inv(theta, splitdata):
    
    zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = get_fullparam(theta)
    
    var_lv_x = ((splitdata['y_lv_err']/slope_lv)**2 + 
             splitdata['x_lv_err']**2 +
             sigma_lv**2)
    model_lv_x = (splitdata['y_lv'] - zeropoint)/slope_lv 
    logl_lv = -0.5 * (np.sum(np.log(2 * np.pi * var_lv_x) + 
                            ((splitdata['x_lv'] - model_lv_x)**2 / var_lv_x) ))
 
    var_hv_x = ((splitdata['y_hv_err']/slope_hv)**2 + 
             splitdata['x_hv_err']**2 +
             sigma_hv**2)
    model_hv_x = (splitdata['y_hv'] - zeropoint)/slope_hv + offset
    logl_hv = -0.5 * (np.sum(np.log(2 * np.pi * var_hv_x) + 
                            ((splitdata['x_hv'] - model_hv_x)**2 / var_hv_x) ))
        
    return logl_lv + logl_hv
    
    
def log_posterior_inv(theta, splitdata):
    
    logpr = log_prior(theta)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood_inv(theta, splitdata)

def log_likelihood_orthogonal(theta, splitdata):
    
    zeropoint, slope_lv, offset, sigma_lv, slope_hv, sigma_hv = get_fullparam(theta)
    
    var_lv = (splitdata['y_lv_err']**2 + 
             (slope_lv*splitdata['x_lv_err'])**2 +
             sigma_lv**2) / (1 + slope_lv**2)
    model_lv = zeropoint + slope_lv * splitdata['x_lv']
    residsq_lv = (splitdata['y_lv'] - model_lv)**2 / (1 + slope_lv**2)
    logl_lv = -0.5 * (np.sum(np.log(2 * np.pi * var_lv) + (residsq_lv / var_lv) ))
 
    var_hv = (splitdata['y_hv_err']**2 + 
             (slope_hv*splitdata['x_hv_err'])**2 +
             sigma_hv**2) / (1 + slope_hv**2)      
    model_hv = zeropoint + slope_hv * (splitdata['x_hv'] - offset)
    residsq_hv = (splitdata['y_hv'] - model_hv)**2 / (1 + slope_hv**2)
    logl_hv = -0.5 * (np.sum(np.log(2 * np.pi * var_hv) + (residsq_hv / var_hv) ))
       
    return logl_lv + logl_hv

    
def log_posterior_orthogonal(theta, splitdata):
    
    logpr = log_prior(theta)
    
    if logpr == -np.inf:
        return logpr
    else:
        return logpr + log_likelihood_orthogonal(theta, splitdata)


    
    
def doMCMC(splitdata, guess, scale, nwalkers=100, nburn=1500, nsteps=3000, mode='Norm'):
    '''
    Takes data which contains color and shape corrected magnitude data
    and performs an mcmc fit on it
    '''
    ndim = len(guess)
    assert ndim == len(scale)

    starting_guesses = np.random.randn(nwalkers, ndim)*scale + guess

    print('sampling...')
    if mode == 'Norm':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads=1, args=[splitdata])
    elif mode == 'inv':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_inv, threads=1, args=[splitdata])
    elif mode == 'ortho':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_orthogonal, threads=1, args=[splitdata])
    elif mode == 'mod':
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_mod, threads=1, args=[splitdata])
    else:
        raise ValueError('mode must be of following: Norm, inv, ortho')

    sampler.run_mcmc(starting_guesses, nsteps)
    print('done')
    
    tlabels = [r"zeropoint", 
           r"slope",
           r"HV offset",
           r"sigma",
           r"HV slope",
           r"HV sigma" ]
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    sampler.reset()

    figcorner = corner.corner(samples, labels=tlabels[0:ndim],
                    show_titles=True, title_fmt=".3f", verbose=True,
                    title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 14})

    return samples


def dolinmix_old(splitdata, offsets, miniter=5000):
    '''
    runs Kelly linmix with offsets
    '''
    n = len(offsets)
    zeropoints = np.zeros(n)
    slopes = np.zeros(n)
    scatters = np.zeros(n)
    
    minscatter = np.inf
    
    y = np.concatenate([splitdata['y_lv'],splitdata['y_hv']])
    xerr = np.concatenate([splitdata['x_lv_err'],splitdata['x_hv_err']])
    yerr = np.concatenate([splitdata['y_lv_err'],splitdata['y_hv_err']])

    # run LinMix for all of the input offsets, save the one with lowest extra scatter needed
    for i in range(n):       
        x = np.concatenate([splitdata['x_lv'],splitdata['x_hv']-offsets[i]])

        lx = linmix.LinMix(x,y,xsig=xerr,ysig=yerr)
        lx.run_mcmc(miniter=miniter,silent=True)

        zeropoints[i] = np.median(lx.chain[:]['alpha'])
        slopes[i] = np.median(lx.chain[:]['beta'])
        scatters[i] = np.sqrt(np.median(lx.chain[:]['sigsqr']))
        
        if scatters[i] < minscatter:
            bestoffset = offsets[i]
            bestlx = copy.copy(lx)
            minscatter = scatters[i]
            
    # calculate the chisq value for each offset without the extra scatter
    inverrsq = np.empty(n)      # variance from data only
    invtoterrsq = np.empty(n)   # variance from data + extra scatter
    for i in range(n):
        inverrsq[i] = np.sum(1.0/(yerr**2 + (slopes[i]*xerr)**2))
        invtoterrsq[i] = np.sum(1.0/(yerr**2 + (slopes[i]*xerr)**2 + scatters[i]**2))

    ndata = len(y)
    residsq = ndata/invtoterrsq # sum of squared residuals with extra scatter gives chisq ~ ndata
    chisq = residsq*inverrsq    # chisq without the extra scatter
    
    poffset = np.exp((np.min(chisq) - chisq)/2.0)  # p(offset) ~ exp(-chisq/2)
    poffset /= np.trapz(poffset,offsets)  # normalize integral to unity

    offsetmean = np.trapz(offsets*poffset,offsets) # calculate pdf mean and stddev
    offsetstddev  = np.sqrt(np.trapz(offsets*offsets*poffset,offsets) - offsetmean**2)
            
    plt.plot(offsets,scatters,'o')
    plt.xlabel('offset')
    plt.ylabel('extra scatter')
    plt.title('offset = {:.3f} +/- {:.3f}'.format(offsetmean, offsetstddev))
    plt.show()
    
    print('offset = {:.5f} +/- {:.5f}'.format(offsetmean, offsetstddev))
       
    return zeropoints, slopes, scatters, bestlx, bestoffset, poffset, offsetmean, offsetstddev

def dolinmix(splitdata, offsets, numiter=5000, mode='forward', returnallsamples=False, downsample=5):
    '''
    runs Kelly linmix with offsets
    '''
    n = len(offsets)
    
    y = np.concatenate([splitdata['y_lv'],splitdata['y_hv']])
    xerr = np.concatenate([splitdata['x_lv_err'],splitdata['x_hv_err']])
    yerr = np.concatenate([splitdata['y_lv_err'],splitdata['y_hv_err']])

    allsamples = np.empty((n,numiter,4))
    
    # run LinMix for all of the input offsets
    print('sampling offsets...')
    for i in range(n):       
        x = np.concatenate([splitdata['x_lv'],splitdata['x_hv']-offsets[i]])

        if mode == 'forward':
            lx = linmix.LinMix(x,y,xsig=xerr,ysig=yerr)
        elif mode == 'backward':
            lx = linmix.LinMix(y,x,xsig=yerr,ysig=xerr)  # fit x on y
        else:
            raise ValueError('mode should be forward or backward')
        
        # run the mcmc
        
        halfiter = numiter // 2  # for some reason we need to divide by 2
        lx.run_mcmc(miniter=halfiter,maxiter=halfiter,silent=True)
       
        assert len(lx.chain[:]['alpha']) == numiter, "unusual number of linmix iterations"
    
        if mode == 'backward':   # we need to invert the parameters to y on x
            fwdalpha = -lx.chain[:]['alpha']/lx.chain[:]['beta']
            fwdbeta = 1.0/lx.chain[:]['beta']
            fwdsigsqr = lx.chain[:]['sigsqr']/lx.chain[:]['beta']**2
            lx.chain[:]['alpha'] = fwdalpha
            lx.chain[:]['beta'] = fwdbeta
            lx.chain[:]['sigsqr'] = fwdsigsqr
        
        if i == 0:
            doffset = (offsets[i+1] - offsets[i])/2.0
        elif i == n-1:
            doffset = (offsets[i] - offsets[i-1])/2.0
        else:
            doffset = (offsets[i+1] - offsets[i-1])/4.0
        
        allsamples[i,:,0] = lx.chain[:]['alpha']
        allsamples[i,:,1] = lx.chain[:]['beta']   
        # smear out the offsets a tiny bit for plotting purposes
        allsamples[i,:,2] = offsets[i] + np.random.uniform(-1,1,size=numiter)*doffset
        allsamples[i,:,3] = np.sqrt(lx.chain[:]['sigsqr'])
        

    print('done')

    scatters = np.average(allsamples[:,:,3],axis=1)
    slopes = np.average(allsamples[:,:,1],axis=1)

    # calculate the chisq value for each offset without the extra scatter
    # variance from data only
    inverrsq = np.sum(1.0/(yerr**2 + (slopes[:,np.newaxis]*xerr)**2),axis=1)
    # variance from data + extra scatter
    invtoterrsq = np.sum(1.0/(yerr**2 + (slopes[:,np.newaxis]*xerr)**2 
                              + (scatters[:,np.newaxis])**2),axis=1)
    ndata = len(y)
    residsq = ndata/invtoterrsq # sum of squared residuals with extra scatter gives median chisq ~ ndata
    chisq = residsq*inverrsq    # chisq without the extra scatter
    
    poffset = np.exp((np.min(chisq) - chisq)/2.0)  # p(offset) ~ exp(-chisq/2)
    poffset /= np.trapz(poffset,offsets)  # normalize integral to unity

    weights = np.tile(poffset[:,np.newaxis],numiter).reshape((n,numiter))
    weights /= np.sum(weights)
    
    # reshape the arrays from (n,numiter) to (n*numiter)
    weights = weights.reshape(-1)
    allsamples = allsamples.reshape((-1,4))
    
    tlabels = [r"zeropoint", 
           r"slope",
           r"HV offset",
           r"sigma"]
    
    if returnallsamples:  

        # return all samples, can only be used with weights  
        figcorner = corner.corner(allsamples, labels=tlabels, weights=weights,
                    show_titles=True, title_fmt=".3f", verbose=True,
                    title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 14})        
        return allsamples, weights, poffset

    else:

        # return weighted samples based on probability
        numweights = len(weights)
        weighted = np.random.choice(numweights, size=numweights//downsample, p=weights)
        samples = allsamples[weighted,:]
        # this works as a sample of the full pdf(alpha, beta, offset, sigma)
        # and can be treated just like MCMC samples 
        figcorner = corner.corner(samples, labels=tlabels, 
                show_titles=True, title_fmt=".3f", verbose=True,
                title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 14})
        return samples, poffset



def dolinmix_sep(splitdata, tol=0.1, numiter=10000, mode='forward', output='sparse'):
    '''
    runs Kelly linmix separately for two groups and then finds offsets based on similar slopes.
    '''

    if mode == 'forward':
        lv = linmix.LinMix(splitdata['x_lv'],splitdata['y_lv'],
                           xsig=splitdata['x_lv_err'],
                           ysig=splitdata['y_lv_err'])
        hv = linmix.LinMix(splitdata['x_hv'],splitdata['y_hv'],
                           xsig=splitdata['x_hv_err'],
                           ysig=splitdata['y_hv_err'])        
    elif mode == 'backward':
        lv = linmix.LinMix(splitdata['y_lv'],splitdata['x_lv'],
                           xsig=splitdata['y_lv_err'],
                           ysig=splitdata['x_lv_err'])
        hv = linmix.LinMix(splitdata['y_hv'],splitdata['x_hv'],
                           xsig=splitdata['y_hv_err'],
                           ysig=splitdata['x_hv_err'])        
    else:
        raise ValueError('mode should be forward or backward')
        
    # run the mcmc        
    halfiter = numiter // 2  # for some reason we need to divide by 2
    lv.run_mcmc(miniter=halfiter,maxiter=halfiter,silent=True)
    hv.run_mcmc(miniter=halfiter,maxiter=halfiter,silent=True)
       
    assert len(lv.chain[:]['alpha']) == numiter, "unusual number of linmix iterations"
    assert len(hv.chain[:]['alpha']) == numiter, "unusual number of linmix iterations"
    
    if mode == 'backward':   # we need to invert the parameters to y on x
        lv.chain[:]['alpha'] /= -lv.chain[:]['beta']
        lv.chain[:]['sigsqr'] /= lv.chain[:]['beta']**2
        lv.chain[:]['beta'] = 1.0/lv.chain[:]['beta']

        hv.chain[:]['alpha'] /= -hv.chain[:]['beta']
        hv.chain[:]['sigsqr'] /= hv.chain[:]['beta']**2
        hv.chain[:]['beta'] = 1.0/hv.chain[:]['beta']

    slope_sig = np.std(np.concatenate([hv.chain['beta'],lv.chain['beta']]))
    slope_tol = tol*slope_sig
    idxs = []
    offsets = []
    slopes = []
    samples = []
    
    # Creating a matrix of the difference in slope
    slope_mat = np.abs(hv.chain['beta'][:,np.newaxis] - lv.chain['beta'])
    min_diff = np.amin(slope_mat)
    slope_temp_hv = hv.chain[:]['beta']
    int_temp_hv = hv.chain[:]['alpha']
    sig2_temp_hv = hv.chain[:]['sigsqr']
    slope_temp_lv = lv.chain[:]['beta']
    int_temp_lv = lv.chain[:]['alpha']
    sig2_temp_lv = lv.chain[:]['sigsqr']
    print('sampling slopes...')
    while (min_diff < slope_tol and len(slope_mat) > 0):
        i, j = np.unravel_index(np.argmin(slope_mat), slope_mat.shape)
        idxs.append((i,j))
        slope_avg = np.average([slope_temp_hv[i], slope_temp_lv[j]])
        slopes.append(slope_avg)
        offset = (int_temp_lv[j] - int_temp_hv[i]) / slope_avg
        offsets.append(offset)
        slope_mat[i,:] = 99.99
        slope_mat[:,j] = 99.99
        min_diff = np.amin(slope_mat)
        
        samples.append([offset, slope_avg, 
                        int_temp_lv[j], int_temp_hv[i], 
                        sig2_temp_lv[j], sig2_temp_hv[i]])
        
    offset_med = np.median(offsets)
    offset_std = np.std(offsets)
    slope_med = np.median(slopes)
    slope_std = np.std(slopes)
    
    tlabels  = [r'offset',
            r'$slope_{avg}$',
            r'zeropoint LV',
            r'zeropoint HV',
            r'$\sigma^2 LV$',
            r'$\sigma^2 HV$']
    figure = corner.corner(samples, labels=tlabels, show_titles=True, title_fmt='.3f', 
                       verbose=True, title_kwargs={'fontsize':11}, 
                       label_kwargs={'fontsize':14})
    
    if output=='sparse':
        return np.array(samples)
    elif output=='verbose':
        return np.array(samples), slope_med, slope_std
    elif output=='full':
        return np.array(samples), slope_med, slope_std, slopes
    else:
        raise ValueError("outout must be 'sparse', 'verbose', or 'full'")
    

def plotFit(samples, splitdata, ndim=4, sigma=True, ax=None, xlabel=None, ylabel=None):
    percentiles = np.percentile(samples, [16,84], axis=0)
    zp = np.median(samples[:,0])
    zp_p = percentiles[:,0]
    offset = np.median(samples[:,2])
    offset_p = percentiles[:,2]
    slope_lv = np.median(samples[:,1])
    slope_lv_p = percentiles[:,1]
    slope_hv = slope_lv
    slope_hv_p = slope_lv_p
    if sigma:
        sigma = np.median(samples[:,3])
    if ndim == 6:
        slope_hv = np.median(samples[:,4])
        slope_hv_p = percentiles[:,4]
        sigma_hv = np.median(samples[:,5])
    if ndim == 4 and not sigma:
        slope_hv = np.median(samples[:,3])
        slope_hv_p = percentiles[:,3]
    
    fit_lv = lambda a: zp + a*slope_lv
    fit_hv = lambda b: zp + (b - offset)*slope_hv
    
    res_lv = fit_lv(splitdata['x_lv']) - splitdata['y_lv']
    res_hv = fit_hv(splitdata['x_hv']) - splitdata['y_hv']
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(splitdata['x_lv'],splitdata['y_lv'],
                xerr=splitdata['x_lv_err'],yerr=splitdata['y_lv_err'],
                linestyle='',ecolor='b',mfc='b',alpha=0.6)
    ax.errorbar(splitdata['x_hv'],splitdata['y_hv'],
                xerr=splitdata['x_hv_err'],yerr=splitdata['y_hv_err'],
                linestyle='',ecolor='r',mfc='r',alpha=0.6)
    
    x_range = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1])
    ax.plot(x_range, fit_lv(x_range),'b-')
    ax.plot(x_range, fit_hv(x_range),'r-')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)

    ax.invert_yaxis()

def plotFit_sep(samples, splitdata, ax=None, xlabel=None, ylabel=None, right=False):
    percentiles = np.percentile(samples, [16,84], axis=0)
    offset = np.median(samples[:,0])
    offset_p = percentiles[:,0]
    slope = np.median(samples[:,1])
    slope_p = percentiles[:,1]
    zp_lv = np.median(samples[:,2])
    zp_lv_p = percentiles[:,2]
    zp_hv = np.median(samples[:,3])
    zp_hv_p = percentiles[:,3]

    fit_lv = lambda a: zp_lv + a*slope
    fit_hv = lambda b: zp_lv + (b - offset)*slope
    
    lfs = 20 # label font size
    tfs = 16 # tick label fontsize
    ebalpha = 0.6 # alpha opacity for errorbars
    linealpha = 0.8 # alpha opacity for fit lines

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.errorbar(splitdata['x_lv'],splitdata['y_lv'],
                xerr=splitdata['x_lv_err'],yerr=splitdata['y_lv_err'],
                linestyle='',ecolor='b',mfc='b',alpha=ebalpha)
    ax.errorbar(splitdata['x_hv'],splitdata['y_hv'],
                xerr=splitdata['x_hv_err'],yerr=splitdata['y_hv_err'],
                linestyle='',ecolor='r',mfc='r',alpha=ebalpha)
    
    xlim_orig = ax.get_xlim()
    ylim_orig = ax.get_ylim()
    x_range = np.linspace(-100,100)
#     x_range = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1])
    
    ax.plot(x_range, fit_lv(x_range),'b-', alpha=linealpha)
    ax.plot(x_range, fit_hv(x_range),'r-', alpha=linealpha)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=lfs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=lfs)
        if right:
            ax.set_ylabel(ylabel, fontsize=lfs, rotation=-90,labelpad=24)
        
    if right:
        ax.yaxis.set_label_position("right")
        ax.tick_params(direction='in',bottom=True,left=True,right=True,top=True,labelsize=tfs,labelright=True)

    ax.set_xlim(xlim_orig[0],xlim_orig[1])
    ax.set_ylim(ylim_orig[0],ylim_orig[1])
    ax.invert_yaxis()