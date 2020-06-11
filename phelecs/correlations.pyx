# cython: language_level=3, boundscheck=False,convert_range=True, wraparound=True
# cython: embedsignature=True

from warnings import warn

cimport numpy as np
import numpy as np

import pandas as pd

from libc.math cimport lrint, floor, fmod, remainder, log, fabs

from cython cimport boundscheck
from cython.parallel import prange
cimport openmp
 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# FREE-RUNNING DATA
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

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
        long len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2
        double l1, l2
        int iBin, curBin
        double[:,:,:] corr = np.zeros([128, 128,
                                    reps], )
        double tmax = <double> times[-1]
        double fstep = <double> step
        double lambdatau = (tmax-lo) / fstep

        double[:] lambdas = np.zeros(128)
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(lo, lo+step*reps+1, step) 

    
    for j in range(max_diff):
        for i in range(len_times -(j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1

            if (diff < binMax) and (diff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = (diff-lo) // step
#==>            try:
                corr[p1, p2, iBin] += 1
#==>            except:
#==>                print(diff, iBin)
            
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


@boundscheck(False)
cpdef crosscorr_freePar(long[:] times,
                        short[:] pos,
                        int lo, 
                        int step,
                        int reps,
                        int max_diff,
                        str norm='lambda'):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement, parallel version.
    
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
        long len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2
        int iThrd
        double l1, l2
        int iBin, curBin
        double[:,:,:]   corr  = np.zeros([128, 128, reps], )
        double[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    128, 128,
                                    reps], )
        double tmax = <double> times[-1]
        double fstep = <double> step
        double lambdatau = (tmax-lo) / fstep

        double[:] lambdas = np.zeros(128)
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(lo, lo+step*reps+1, step) 

    
    for j in prange(max_diff, nogil=True):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times- (j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1

            if (diff < binMax) and (diff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = (diff-lo) // step

                pcorr[iThrd, p1, p2, iBin] += 1

    for p1 in range(128):
        for p2 in range(128):
            for j in range(reps):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]
            
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

@boundscheck(False)
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
        long len_times, i, j
        long t1, t2, diff, 
        double binMax
        int p1, p2
        int iThrd
        int iBin, curBin
        double[:,:,:] corr = np.zeros([128, 128,
                                    int(no_bins)], )
        double[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    128, 128,
                                    int(no_bins)], )
        
        double tmax = <double> times[-1]
        double step = (hi-lo) / no_bins
        double logdiff
        
        double[:] lambdas = np.zeros(128)

    binMax = hi #- step
    
    len_times = len(times)
    bins = np.exp( np.linspace(lo,        
                               hi,       
                               int(no_bins+1)) )   
    normfac = tmax / np.diff(bins)                
    normfac -= bins[:-1] / np.diff(bins)           
    
    for j in prange(max_diff,
                    nogil=True):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times- (j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1
            logdiff = log(<double> diff)

            if (logdiff < binMax) and (logdiff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = <int>( (logdiff-lo) / step )
                pcorr[iThrd, p1, p2, iBin] += 1

    for p1 in range(128):
        for p2 in range(128):
            for j in range(int(no_bins)):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]

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

# from libc.stdio cimport printf


cpdef crosscorr_free_binnedPar(long[:] times,
                               short[:] pos,
                               long reps,
                               double width,
                               int max_diff,
                               double cycle=47856.62402159,
                               norm='lambda'):
    """Binned auto- and cross-correlations for a free-running
    photoelectron measurement, parallel version.
    
    Parameters
    ----------
    times    : Event arrival times.
    pos      : Corresp. detector positions.
    reps     : Number of bins. 
    width    : Width of the integration region `[-width, +width]`
    cycle    : Duration of a bin in units of `times`. Maximum lag time is reps * cycle.
    max_diff : Maximum number of forward differences `t_{i+max_diff} - t_{i}`
               that are formed for the correlation calculation.
    norm     : One of `lambda`, `lamdatau`, `one` or `none`.

    Returns
    -------

    corr : 3D array of dimension (128, 128, reps). Contains the auto- (on-axis)
           and cross-correlations (off-axis). The default normalization,
           `lambda`, is chosen such that, for a Poissonian process of likelihood
           lambda, the autocorrelation is lambda**2.
"""
    cdef:
        long len_times, i, j
        long t1, t2, binMax
        double diff, fstep, remain
        int p1, p2
        int iThrd, iBin
        double[:,:,:] corr = np.zeros([128, 128,
                                    reps], )
        int[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    128, 128,
                                    reps], dtype=np.int32)
        

        double tmax = <double> times[-1]
        double lambdatau = tmax / cycle

        double[:] lambdas = np.zeros(128)
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(reps) * cycle

    binMax = <long> ( (reps-1) * cycle + width ) 

#   print('lambdatau is ', lambdatau)

    for j in prange(max_diff, 
            nogil=True):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times- (j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = <double>( t2 - t1 )

            if diff <= binMax: 

                remain = remainder(diff, cycle)

                if fabs(remain) < width:
                    p1 = pos[i]
                    p2 = pos[i+j+1]
                    iBin = <int>( (diff+width) / cycle )

    #               try:
                    pcorr[iThrd, p1, p2, iBin] += 1
    #               except:
    #                   print(diff, remain, iBin)
    
    for p1 in range(128):
        for p2 in range(128):
            for j in range(reps):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]
            

    if norm=='none':
        return np.asarray(corr), bins

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins

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

        return np.asarray(corr) / (normfac * cycle**2), bins



    elif norm=='lambda':
        return np.asarray(corr) / (normfac * cycle**2), bins

    else:
        warn('Unclear choice of normalization. Using the default `lambda` instead.')
        return np.asarray(corr) / (normfac * cycle**2), bins

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

@boundscheck(False)
cpdef tune_bin_freqPar(long[:] times,
                    double width,
                    int no_bins,
                    int max_diff,
                    double cycle=47856.62402159,
        ):
    cdef:
        long len_times = len(times)
        long i, j
        long t1, t2, binMax
        double diff, fstep, remain
        int p1, p2
        int iThrd
        double[:]   corr  = np.zeros(no_bins)
        double[:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                      no_bins])

    bins = np.linspace(-width, width, no_bins+1)
    fstep = (bins[1] - bins[0])
    bins = (bins[1:] + bins[:-1]) / 2

    for j in prange(max_diff, nogil=True):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times- (j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = <double>( t2 - t1 )

#           if diff < width: continue

            remain = remainder(diff, cycle)

            if (remain >= -width) and (remain < (width-fstep) ):
                p1 = <int>( (remain + width) / fstep )
                pcorr[iThrd, p1] += 1

    for i in range(openmp.omp_get_max_threads()):
        for j in range(no_bins):
            corr[j] += pcorr[i, j]
  

    return  np.asarray(corr), bins # / np.asarray(norm)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

@boundscheck(False)
cpdef tune_bin_stdPar(long[:] times,
                    double width,
                    int no_bins,
                    int max_diff,
                    double cycle=47856.643,
        ):
    cdef:
        long len_times = len(times)
        long i, j
        long t1, t2, binMax
        double diff, fstep, remain, b
        int p1, p2
        int iThrd, nThrd = openmp.omp_get_max_threads()
        double[:] std   = np.zeros(nThrd)
        double[:] pcorr = np.zeros(nThrd)
        double[:] norm  = np.zeros(nThrd)
#       double[:] cbins

    bins = np.linspace(-width, width, no_bins+1)
    fstep = (bins[1] - bins[0])
    bins = (bins[1:] + bins[:-1]) / 2
#   cbins = bins

    for j in prange(max_diff, nogil=True,
                    schedule='dynamic'):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times-(j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = <double>( t2 - t1 )

            if diff < width: continue

            remain = remainder(diff, cycle)

            if (remain >= -width) and (remain < (width-fstep) ):
#               p1 = <int>( (remain + width) / fstep )
#               b = cbins[p1] # This one's the problem

#               pcorr[iThrd] += remain
#               std[iThrd]   += remain * remain
                norm[iThrd]  += 1



    return  #np.asarray(pcorr).sum(), np.asarray(std).sum(), np.asarray(norm).sum()
  

cpdef crosscorr_narrow(long[:] times,
                       long lo, 
                       long reps,
                     int min_diff,
                     int max_diff,
                     ):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement.
    
    Parameters
    ----------
    times    : Event arrival times
    lo       : Smallest lag time on the lag-time axis.
    reps     : Number of steps. Maximum lag time is thererfore `lo + step * reps`.

    Returns
    -------

    corr : 1D array of dimension (reps). 
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        long len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2
        int iBin, curBin
        double[:] corr = np.zeros([reps], )

    
        
    binMax = lo + reps
    len_times = len(times)

    bins = np.arange(reps) + lo

    
    for i in range(len_times-max_diff):
        t1 = times[i]
        for j in range(min_diff, max_diff - 1,
                       1):
            t2 = times[i+j+1]
            diff = t2 - t1

            if diff >= binMax: continue
            if (diff < binMax) and (diff >= lo):
                iBin = (diff-lo)
#==>            try:
                corr[iBin] += 1
#==>            except:
#==>                print(diff, iBin)

            
    return np.asarray(corr), bins


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# TRIGGERED DATA
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cpdef crosscorr_trigd(long[:] startCtr,
                      long[:] bunches,
                      short[:] pos,
                      long no_cycles, 
                      int max_diff,
                      norm='lambda',
                      double cycle=47856.62402159):
    """Full auto- and cross-correlations for a photoelectron measurement
       synchronized to an external start trigger.
    
    Parameters
    ----------
    startCtr : Start counter.
    bunches  : Bunch indices for every start counter.
    pos      : Detector position.
    no_cycles: Number of cycles over which to correlate.
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
        long len_times, i, j
        long t1, t2, c1, c2, diff, binMax
        int p1, p2
#         unsigned int iBin, curBin
#         unsigned long[:]  bins = np.zeros(no_bins, dtype=np.uint64)
        int iThrd 
        long cDif
        double[:,:,:] corr = np.zeros([128, 128,
                                          no_cycles*24], )
        int[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    128, 128,
                                    no_cycles*24], dtype=np.int32)
        double lambdatau = startCtr[-1] * 24
        double tmax = (lambdatau + bunches[-1]) * cycle

        double[:] lambdas = np.zeros(128)
        
    
    normfac = lambdatau - np.arange(no_cycles*24, 
                                    dtype=np.float_)

    len_times = len(bunches)
    bins = np.arange(no_cycles*24) * cycle
    binMax = (no_cycles * 24)
    
    for j in prange(max_diff, 
                    nogil=True,):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times - (j+1) ):
            t1 = bunches[i]
            c1 = startCtr[i]

            t2 = bunches[i+j+1]
            c2 = startCtr[i+j+1]
#             t2 = t2 + (24 * c2)
            
            p1 = pos[i]
            p2 = pos[i+j+1]

            cDif = c2 - c1
            
            if c1 == c2:
                diff = max(t1, t2) - min(t1, t2)
                pcorr[iThrd, p1, p2, diff] += 1
            elif (cDif > 0) and (cDif < no_cycles):
                diff = cDif*24 +t2 - t1
                pcorr[iThrd, p1, p2, diff] += 1

#                 diff = 
#             if diff < binMax:

            
    for p1 in range(128):
        for p2 in range(128):
            for j in range(no_cycles*24):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]

    if norm=='none':
        return np.asarray(corr), bins

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins

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

                for i in range(no_cycles*24):
                    corr[p1,p2,i] /= (l1 * l2) / tmax

        return np.asarray(corr) / (normfac * cycle**2), bins



    elif norm=='lambda':
        return np.asarray(corr) / (normfac * cycle**2), bins

    else:
        warn('Unclear choice of normalization. Using the default `lambda` instead.')
        return np.asarray(corr) / (normfac * cycle**2), bins

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# PRE-ACCUMULATED DATA
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cpdef cumCrossCorrLin(long[:]   times,
                      short[:]  pos,
                      double[:] count,
                        int lo, 
                        int step,
                        int reps,
                        int max_diff,
                        str norm='lambda',
                        int npix=128,
                        ):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement, parallel version.
    
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
    npix     : Width of the detector in pixels.

    Returns
    -------

    corr : 3D array of dimension (npix, npix, reps). Contains the auto- (on-axis)
           and cross-correlations (off-axis). The default normalization,
           `lambda`, is chosen such that, for a Poissonian process of likelihood
           lambda, the autocorrelation is lambda**2.
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        long len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2, 
        double s1, s2
        int iThrd
        double l1, l2
        int iBin, curBin
        double[:,:,:]   corr  = np.zeros([npix, npix, reps], )
        double[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    npix, npix,
                                    reps], )
        double tmax = <double> times[-1]
        double fstep = <double> step
        double lambdatau = (tmax-lo) / fstep

        double[:] lambdas = np.zeros(npix)
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(lo, lo+step*reps+1, step) 

    
    for j in prange(max_diff, nogil=True):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times- (j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1

            if (diff < binMax) and (diff >= lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = (diff-lo) // step

                s1 = count[i]
                s2 = count[i+j+1]

                pcorr[iThrd, p1, p2, iBin] += s1 * s2

    for p1 in range(npix):
        for p2 in range(npix):
            for j in range(reps):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]
            
    if norm=='none':
        return np.asarray(corr), bins[:-1]

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins[:-1]

    elif norm=='one':

        tmax = tmax*tmax

        for i in range(len_times):
            p1 = pos[i]
            s1 = count[i]
            lambdas[p1] += s1

        for p1 in range(npix):
            l1 = lambdas[p1]
            if not l1: continue

            for p2 in range(npix):
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
# PRE-ACCUMULATED DATA IN TWO DIMENSIONS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

cpdef cumCrossCorrLin2D(long[:]   times,
                        double[:,:] count,
                        int lo, 
                        int step,
                        int reps,
                        int max_diff,
                        str norm='lambda',
                        ):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement, parallel version.
    
    Parameters
    ----------
    times    : Event arrival times
    count    : Corresp. spectra, 2D array of shape (npix, len(times))
    lo       : Smallest lag time on the lag-time axis.
    step     : Step size of the lag-time axis.
    reps     : Number of steps. Maximum lag time is thererfore `lo + step * reps`.
    max_diff : Maximum number of forward differences `t_{i+max_diff} - t_{i}`
               that are formed for the correlation calculation.
    norm     : One of `lambda`, `lamdatau`, `one` or `none`.

    Returns
    -------

    corr : 3D array of dimension (npix, npix, reps). Contains the auto- (on-axis)
           and cross-correlations (off-axis). The default normalization,
           `lambda`, is chosen such that, for a Poissonian process of likelihood
           lambda, the autocorrelation is lambda**2.
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        long len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2, 
        double s1, s2
        int iThrd
        double l1, l2
        int iBin, curBin
        int npix = count.shape[1]
        double[:,:,:]   corr  = np.zeros([npix, npix, reps], )
        double[:,:,:,:] pcorr = np.zeros([openmp.omp_get_max_threads(),
                                    npix, npix,
                                    reps], )
        double tmax = <double> times[-1]
        double tmin = <double> times[ 0]
        double fstep = <double> step
        double lambdatau = (tmax-tmin) / fstep 

        double[:] lambdas = np.zeros(npix)
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    normfac = lambdatau - np.arange(reps, dtype=np.float_)
    bins = np.arange(lo, lo+step*reps+1, step) 

    for j in prange(max_diff, nogil=True,
                    schedule='static'):
#   for j in  range(max_diff,           ):
        iThrd = openmp.omp_get_thread_num()
        for i in range(len_times-(j+1)):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1

            if (diff < binMax) and (diff >= lo):

                iBin = (diff-lo) // step


                for p1 in range(npix):
                    s1 = count[i,p1]
                    for p2 in range(npix):
                        s2 = count[i+j+1,p2]

                        pcorr[iThrd, p1, p2, iBin] += s1 * s2


    for p1 in range(npix):
        for p2 in range(npix):
            for j in range(reps):
                for i in range(openmp.omp_get_max_threads()):
                    corr[p1,p2,j] += pcorr[i,p1,p2,j]
            
    if norm=='none':
        return np.asarray(corr), bins[:-1]

    elif norm=='lambdatau':
        return np.asarray(corr) / (normfac), bins[:-1]

    elif norm=='one':

        tmax = tmax*tmax

        for i in range(len_times):
            for p1 in range(npix):
                s1 = count[i,p1]
                lambdas[p1] += s1

        for p1 in range(npix):
            l1 = lambdas[p1]
            if not l1: continue

            for p2 in range(npix):
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
