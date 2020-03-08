
cimport numpy as np
import numpy as np

from libc.math cimport lrint, floor, fmod, remainder, log
 
cpdef crosscorr_free(long[:] times,
                    short[:] pos,
                    int lo, 
                    int step,
                    int reps,
                    int max_diff):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement.
    
    Parameters
    ----------
    times   : Event arrival times
    pos     : Corresp. detector positions
    lo      : Smallest lag time on the lag-time axis.
    step    : Step size of the lag-time axis.
    reps    : Number of steps. Maximum lag time is thererfore `lo + step * reps`.
    max_diff: Maximum number of forward differences `t_{i+max_diff} - t_{i}` that are formed
                for the correlation calculation.

    Returns
    -------

    corr : 3D array of dimension (128, 128, reps). Contains the auto- (on-axis) and cross-correlations (off-axis). The normalization is chosen such that, for a Poissonian process of likelihood lambda, the autocorrelation is lambda**2.
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""
    cdef:
        int len_times, i, j
        int t1, t2, diff, binMax
        int p1, p2
        int iBin, curBin
        int[:,:,:] corr = np.zeros([128, 128,
                                    reps], dtype=np.int32)
        double lambdatau = <double> times[-1] / step
    
        
    binMax = lo + step * reps
    
    len_times = len(times)

    norm = lambdatau - np.arange(reps, dtype=np.float_)
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
            
    return np.asarray(corr) / (norm * step**2), bins[:-1]




cpdef crosscorr_free_log(long[:] times,
                         short[:] pos,
                         double lo, 
                         double hi, 
                         double no_bins,
                         int max_diff):
    cdef:
        int len_times, i, j
        int t1, t2, diff, 
        double binMax
        int p1, p2
        int iBin, curBin
        int[:,:,:] corr = np.zeros([128, 128,
                                    int(no_bins)], dtype=np.int32)
        
        double step = (hi-lo) / no_bins
        double logdiff

    binMax = hi - step
    
    len_times = len(times)
    bins = np.exp( np.linspace(lo, hi, int(no_bins+1)) )
    norm = times[-1] / np.diff(bins)
    
    for j in range(max_diff):
        for i in range(len_times-max_diff):
            t1 = times[i]
            t2 = times[i+j+1]
            diff = t2 - t1
            logdiff = log(<double> diff)

            if (logdiff < binMax) and (logdiff > lo):
                p1 = pos[i]
                p2 = pos[i+j+1]
           
                iBin = lrint( (logdiff-lo) / step)
                corr[p1, p2, iBin] += 1
            
    return np.asarray(corr) / (norm * np.diff(bins)**2), bins[:-1]



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


cpdef tune_bin_freq(long[:] times,
                    double[:] reps,
                    long width,
                    double cycle=47856.643,
                    int max_diff=100,

        ):
    cdef:
        int len_times, i, j
        long t1, t2, binMax
        long diff
        int p1, p2
        int L
        long[:] lo_bin 
        double[:] corr = np.zeros_like(reps, dtype=np.float_)
        double[:] norm = np.zeros_like(reps, dtype=np.float_)

    lo_bin = np.rint(np.asarray(reps) * cycle - width).astype(np.int_)
                     
    binMax = lo_bin[-1] + 2 * width
        
    len_times = len(times)
    
    for i in range(len_times-max_diff):
        L = 0
        t1 = times[i]
        
        for j in range(max_diff-1):
            t2 = times[i+j+1]
            diff = t2 - t1

            if diff > binMax: 
                continue
            else:
                pass

            if (L >= 0) and (diff < (lo_bin[L] + 2*width)):
                corr[L] += <double> diff - lo_bin[L] - width
                norm[L] += 1
            elif L < 0: L=0
            
    return  np.asarray(corr) # / np.asarray(norm)
