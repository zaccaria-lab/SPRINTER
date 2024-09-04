from utils import *

from numba import jit



def calc_rt_rdrs(counts, rt_size, maxgap=20, min_frac_bins=.1, min_bins=4, visual=False, j=1):
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=j, 
                  initializer=init_calc_rt_rdrs, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in counts.groupby('CELL')}),
                            shared,
                            manager.dict(rt_size.to_dict()),
                            maxgap,
                            min_frac_bins,
                            min_bins,
                            visual)) \
        as pool:
            bar = ProgressBar(total=counts['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(run_calc_rt_rdrs, counts['CELL'].unique()) if progress(cell)]
        return pd.concat(shared)\
                 .sort_values(['CELL', 'CHR', 'START', 'END'])\
                 .reset_index(drop=True)
    

def init_calc_rt_rdrs(_counts, _shared, _rt_size, _maxgap, _min_frac_bins, _min_bins, _visual):
    global COUNTS, SHARED, RT_SIZE, MAXGAP, MIN_FRAC_BINS, MIN_BINS, VISUAL
    COUNTS = _counts
    SHARED = _shared
    RT_SIZE = _rt_size
    MAXGAP = _maxgap
    MIN_FRAC_BINS = _min_frac_bins
    MIN_BINS = _min_bins
    VISUAL = _visual


def run_calc_rt_rdrs(cell, minrdr=0., maxrdr=3.):
    counts = COUNTS[cell].sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)    
    counts['FORCE_BK'] = (counts.groupby('consistent_rt', sort=False)['CHR'].transform(lambda x : x != x.shift(1)) | \
                          counts.groupby('consistent_rt', sort=False)['RANK'].transform(lambda x : (x - x.shift(1)) > MAXGAP)).astype(int).cumsum()

    minp = min(MIN_BINS, max(1, int(np.ceil(MIN_FRAC_BINS * RT_SIZE[cell]))))

    counts['RAW_GC'] = counts['GC'].values
    counts['GC'] = counts[::-1].groupby(['FORCE_BK', 'consistent_rt'], sort=False)['GC']\
                               .rolling(RT_SIZE[cell], min_periods=1, center=False)\
                               .mean().droplevel([0, 1])[::-1].sort_index()

    counts['WIND_COUNT'] = counts[::-1].groupby(['FORCE_BK', 'consistent_rt'], sort=False)['COUNT']\
                                       .rolling(RT_SIZE[cell], min_periods=minp, center=False)\
                                       .sum().droplevel([0, 1])[::-1].sort_index()
    counts['WIND_NORM_COUNT'] = counts[::-1].groupby(['FORCE_BK', 'consistent_rt'], sort=False)['NORM_COUNT']\
                                            .rolling(RT_SIZE[cell], min_periods=minp, center=False)\
                                            .sum().droplevel([0, 1])[::-1].sort_index()
    counts['RAW_RDR'] = counts['WIND_COUNT'] / counts['WIND_NORM_COUNT']
    counts['RAW_RDR'] = counts['RAW_RDR'] / counts['RAW_RDR'].mean()

    counts['RAW_RDR'] = counts.groupby(['CHR', 'consistent_rt'])['RAW_RDR'].transform(prev_else_next_rdrs,
                                                                                      engine='numba',
                                                                                      engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
    
    if not VISUAL:
        counts['RAW_RDR'] = counts.groupby('BIN_REPINF')['RAW_RDR'].transform(boot_rdrs_rt,
                                                                            engine='numba',
                                                                            engine_kwargs={'nopython': True, 'nogil': True, 'cache' : True, 'fastmath' : False})
    
    counts['RAW_RDR'] = counts['RAW_RDR'].clip(lower=minrdr, upper=maxrdr)
    counts['RAW_RDR'] = counts['RAW_RDR'] / counts['RAW_RDR'].mean()
    assert not pd.isnull(counts['RAW_RDR']).any(), 'Found a NaN in RDR estimation'
    SHARED.append(counts.drop(columns=['FORCE_BK']))
    return cell


@jit(nopython=True, fastmath=False, cache=True)
def boot_rdrs_rt(values, index, num_repeats=10):
    res = np.empty(values.shape[0], dtype=np.float64)
    repeats = np.empty(num_repeats, dtype=np.float64)
    sample = np.empty(values.shape[0], dtype=np.float64)
    for pos in range(values.shape[0]):
        for rep in range(repeats.shape[0]):
            for b in range(values.shape[0]):
                sample[b] = values[np.random.randint(0, values.shape[0])]
            repeats[rep] = np.mean(sample)
        res[pos] = np.mean(repeats)
    res[0] = values[0]
    if values.shape[0] > 1:
        mean = res[1:].mean()
        if mean > 0 and (not np.isnan(mean)):
            res[1:] = res[1:] * (res[0] / mean)
    return res


@jit(nopython=True, fastmath=False, cache=True)
def prev_else_next_rdrs(values, index):
    A = values
    n = A.shape[0]
    Aprev = np.empty(n, dtype=A.dtype)
    Anext = np.empty(n, dtype=A.dtype)
    for i in range(n):

        if not np.isnan(A[i]):
            Aprev[i] = A[i]
        else:
            if i == 0:
                Aprev[i] = np.nan
            else:
                Aprev[i] = Aprev[i-1]
        
        j = n - 1 - i
        if not np.isnan(A[j]):
            Anext[j] = A[j]
        else:
            if j == n - 1:
                Anext[j] = np.nan
            else:
                Anext[j] = Anext[j+1]

    res = np.empty(n, dtype=A.dtype)
    for i in range(n):
        if not np.isnan(Aprev[i]):
            res[i] = Aprev[i]
        elif not np.isnan(Anext[i]):
            res[i] = Anext[i]
        else:
            res[i] = np.nan
    
    return res

