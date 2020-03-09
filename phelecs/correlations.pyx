from warnings import warn

cimport numpy as np
import numpy as np

import pandas as pd

from libc.math cimport lrint, floor, fmod, remainder, log
 
cpdef crosscorr_free(long[:] times,
                    short[:] pos,
                    int lo, 
                    int step,
                    int reps,
                    int max_diff,
                    str norm='lambda'):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement.
    
    Parameters
    ----------
    times    : Event arrival times
    pos      : Corresp. detector positions
    lo       : Smallest lag time on the lag-time axis.
    step     : Step size of the lag-time axis.
    reps     : Number of steps. Maximum lag time is thererfore `lo + step * reps`.
    max_diff : Maximum number of forward differences `t_{i+max_diff} - t_{i}`
               that are formed for the correlation calculation.
    norm     : One of `lambda`, `lamdatau`, `one` or `none`.

    Returns
    -------

    corr : 3D array of dimension (128, 128, reps). Contains the auto- (on-axis)
           and cross-correlations (off-axis). The default normalization,
           `lambda`, is chosen such that, for a Poissonian process of likelihood
           lambda, the autocorrelation is lambda**2.
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        int len_times, i, j
        int t1, t2, diff, binMax
        int p1, p2
        double l1, l2
        int iBin, curBin
        double[:,:,:] corr = np.zeros([128, 128,
                                    reps], )
        double tmax = <double> times[-1]
        double fstep = <double> step
        double lambdatau = tmax / fstep

        double[:] lambdas = np.zeros(128)
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(lo, lo+step*reps+1, step) 

    
    for j in range(max_diff):
        for i in range(len_times-max_diff):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1

            if (diff < binMax) and (diff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = (diff-lo) // step
                try:
                    corr[p1, p2, iBin] += 1
                except:
                    print(diff, iBin)
            
    if norm=='none':
        return np.asarray(corr), bins[:-1]

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins[:-1]

    elif norm=='one':

        tmax = tmax*tmax

        for i in range(len_times):
            p1 = pos[i]
            lambdas[p1] += 1

        for p1 in range(128):
            l1 = lambdas[p1]
            if not l1: continue

            for p2 in range(128):
                l2 = lambdas[p2]
                if not l2: continue

                for i in range(reps):
                    corr[p1,p2,i] /= (l1 * l2) / tmax

        return np.asarray(corr) / (normfac * step**2), bins[:-1]



    elif norm=='lambda':
        return np.asarray(corr) / (normfac * step**2), bins[:-1]

    else:
        warn('Unclear choice of normalization. Using the default `lambda` instead.')
        return np.asarray(corr) / (normfac * step**2), bins[:-1]


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


cpdef crosscorr_free_log(long[:] times,
                         short[:] pos,
                         double lo, 
                         double hi, 
                         double no_bins,
                         int max_diff,
                         str norm='lambda'):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement on a logarithmic scale.
    
    Parameters
    ----------
    times    : Event arrival times
    pos      : Corresp. detector positions
    lo       : Smallest lag time on the lag-time axis.
    hi       : Largest lag time on the lag-time axis.
    no_bins  : Number of bins along the lag-time axis.
    max_diff : Maximum number of forward differences `t_{i+max_diff} - t_{i}`
               that are formed for the correlation calculation.
    norm     : One of `lambda`, `lamdatau`, `one` or `none`.

    Returns
    -------

    corr : 3D array of dimension (128, 128, no_bins). Contains the auto- (on-axis)
           and cross-correlations (off-axis). The default normalization,
           `lambda`, is chosen such that, for a Poissonian process of likelihood
           lambda, the autocorrelation is lambda**2.
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        int len_times, i, j
        int t1, t2, diff, 
        double binMax
        int p1, p2
        int iBin, curBin
        double[:,:,:] corr = np.zeros([128, 128,
                                    int(no_bins)], )
        double tmax = <double> times[-1]
        double step = (hi-lo) / no_bins
        double logdiff
        
        double[:] lambdas = np.zeros(128)

    binMax = hi #- step
    
    len_times = len(times)
    bins = np.exp( np.linspace(lo,        # I really don't understand
                               hi,        # why I would need the minus half step,
                               int(no_bins+1)) )   # but that's the only way the normalization
    normfac = tmax / np.diff(bins)                 # seems to be accurate...
    normfac -= bins[:-1] / np.diff(bins)           
    
    for j in range(max_diff):
        for i in range(len_times-max_diff):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1
            logdiff = log(<double> diff)

            if (logdiff < binMax) and (logdiff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = <int>( (logdiff-lo) / step )
                corr[p1, p2, iBin] += 1

    if norm=='none':
        return np.asarray(corr), bins[:-1]

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins[:-1]

    elif norm=='one':

        tmax = tmax*tmax

        for i in range(len_times):
            p1 = pos[i]
            lambdas[p1] += 1

        for p1 in range(128):
            l1 = lambdas[p1]
            if not l1: continue

            for p2 in range(128):
                l2 = lambdas[p2]
                if not l2: continue

                for i in range(<int> no_bins):
                    corr[p1,p2,i] /= (l1 * l2) / tmax

        return np.asarray(corr) / (normfac * np.diff(bins)**2), bins[:-1]



    elif norm=='lambda':
        return np.asarray(corr) / (normfac * np.diff(bins)**2), bins[:-1]

    else:
        warn('Unclear choice of normalization. Using the default `lambda` instead.')
        return np.asarray(corr) / (normfac * np.diff(bins)**2), bins[:-1]
            

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cpdef crosscorr_free_binned(long[:] times,
                            short[:] pos,
                            long rep,
                            long width,
                            max_diff=100):
    cdef:
        int len_times, i, j
        long t1, t2, binMax
        long diff
        int p1, p2
        int L
        long[:] lo_bin 
        int[:,:,:] corr = np.zeros([128, 128,
                                    rep], dtype=np.int32)
        
        double cycle = 47856.643
    
    lo_bin = np.rint(np.arange(rep) * cycle - width)
    binMax = lo_bin[-1] + 2 * width
        
    len_times = len(times)
    
    for i in range(len_times-max_diff):
        L = 0
        t1 = times[i]
        p1 = pos[i]
        
        for j in range(max_diff-1):
            t2 = times[i+j+1]
            p2 = pos[i+j+1]
            diff = t2 - t1

            if diff > binMax: 
                continue
            else:
                pass

            if (L >= 0) and (diff < (lo_bin[L] + 2*width)):
                corr[p1, p2, L] += 1
            elif L < 0: L=0
            
    return  np.asarray(corr) 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cpdef tune_bin_freq(long[:] times,
                    long width,
                    int no_bins,
                    double cycle=47856.643,
                    int max_diff=100,

        ):
    cdef:
        int len_times, i, j
        long t1, t2, binMax
        long diff
        double remain
        int p1, p2
        int L
        long[:] lo_bin 
        double[:] corr = np.zeros(no_bins)

    bins = np.linspace(-width, width, no_bins) 
    fstep = <double> (bins[1] - bins[0])

    for i in range(len_times-max_diff):
        L = 0
        t1 = times[i]
        
        for j in range(max_diff-1):
            t2 = times[i+j+1]
            diff = t2 - t1

            remain = remainder(diff, cycle)

            if (remain > -width) and (remain < width):
                p1 = <int>( (remain + width) / fstep )
                corr[p1] += 1

    return  np.asarray(corr) # / np.asarray(norm)
