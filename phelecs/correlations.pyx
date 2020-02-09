
cimport numpy as np
import numpy as np


cdef inline binary_leftmost(long[:] lims,
                            long    val,
                            long L=0):
    cdef long R, m

    R = len(lims) 
    
    while L < R:
        m = (L+R)/2
        if lims[m] > val: 
            R = m
        else: L = m + 1
    return L - 1


cpdef crosscorr_free(long[:] times,
                     short[:] pos,
                     lo, hi, no_bins,
                     max_diff):
    """Full auto- and cross-correlations for a free-running
    photoelectron measurement.
    
    Parameters
    ----------
    times   : event arrival times
    pos     : corresp. detector positions
    lo      : smallest lag time, corresponds to the first entry in the correlation tensor.
    hi      : largest  lag time, last entry on the lag-time axis.
    no_bins : number of bins between `lo` and `hi`.
    max_diff: Maximum number of forward differences t_{i+max_diff} - t_{i} that are formed
                for the correlation calculation.

    Returns
    -------

    corr : 3D array of dimension (128, 128, no_bins). Contains the auto- (on-axis) and cross-correlations (off-axis)
    bins : Time-lag axis, dimension (no_bins,), for plotting `corr`.
"""

    cdef:
        int len_times, i, j
        long t1, t2, diff, binMax
        int p1, p2
        int L
        long[:]  bins = np.zeros(no_bins, dtype=np.int_)
        int[:,:,:] corr = np.zeros([128, 128,
                                   no_bins], dtype=np.int32)
        
    bins = np.linspace(lo, hi, no_bins,
                       dtype=np.int_)
    binMax = bins[-1]
    
    len_times = len(times)
    
    for i in range(len_times-max_diff):
        L = 0
        t1 = times[i]
        p1 = pos[i]
        
        for j in range(max_diff):
            t2 = times[i+j+1]
            p2 = pos[i+j+1]
            diff = t2 - t1
            
            if diff > binMax: 
                continue
            else:
                L = binary_leftmost(bins, diff,
                                    L=L)
            if L >= 0:
                corr[p1, p2, L] += 1
            else: L = 0
            

    return np.asarray(bins), np.asarray(corr) 



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
                L = binary_leftmost(lo_bin, diff,
                                    L=L)

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
                L = binary_leftmost(lo_bin, diff,
                                    L=L)

            if (L >= 0) and (diff < (lo_bin[L] + 2*width)):
                corr[L] += <double> diff - lo_bin[L] - width
                norm[L] += 1
            elif L < 0: L=0
            
    return  np.asarray(corr) # / np.asarray(norm)
