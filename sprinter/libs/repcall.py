from utils import * 

from statsmodels.distributions.empirical_distribution import ECDF
# from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.nonparametric.kde import bandwidths, forrt, revrt, silverman_transform, fast_linbin



def infersphase_local(cell, data, seg, permutations=10000, permethod='permutation', testmethod='density', meanmethod='weighted_min', 
                      timing='consistent_rt', rdr_profile=None, maxrtrdr=3, quantile=.05, toplot=False, histbins=None):
    rdr_profile = 'RT_PROFILE'
    columns = ['CELL', 'CHR', 'START', 'RDR', 'RT_PROFILE', timing, 'GC', 'GENOME']
    columns = columns + [seg] if seg not in columns and seg != 'WHOLE_GENOME' else columns
    celldf = data[columns].drop_duplicates().reset_index(drop=True)
    if len(celldf) == 0:
        return {'CELL' : cell,
                'PVAL_EMP' : 1.0,
                'PVAL_FIT' : 1.0,
                'P_OF_FIT' : 1.0,
                'PVAL_COMB' : 1.0,
                'IS_MAGENTA_HIGHER' : False}
    celldf['WHOLE_GENOME'] = 0
    celldf['MYSEG'] = celldf[seg].map(dict(zip(celldf[seg].unique(), np.arange(celldf[seg].nunique())))).astype(int)
    celldf[rdr_profile] = celldf[rdr_profile].clip(0, maxrtrdr)
    celldf = celldf.sort_values(['MYSEG', rdr_profile], ascending=[True, True]).reset_index(drop=True)
    assert np.all(celldf.groupby(seg)[timing].apply(lambda col : 'early' in col.unique() and 'late' in col.unique())), celldf
    assert np.all(celldf.groupby('MYSEG')[timing].apply(lambda col : 'early' in col.unique() and 'late' in col.unique())), celldf
    assert set(celldf[timing].unique()) == {'early', 'late'}
    seg = celldf['MYSEG'].values
    rep = (celldf[timing].values == 'late')
    rdr = celldf[rdr_profile].values
    if testmethod in ['density', 'overlap', 'integral']:
        celldf['SIGNED_RDR'] = celldf[rdr_profile] * np.where(celldf[timing]=='early', +1, -1)
        binning = (lambda x : np.histogram_bin_edges(x, bins='fd', range=(0, maxrtrdr)))
        choose_bins = (lambda _rdr, _rep : min((binning(_rdr[_rep]), binning(_rdr[~_rep])), key=(lambda hist : len(hist))))
        rdr = celldf.groupby('MYSEG')['SIGNED_RDR'].transform(lambda S : np.digitize(np.abs(S), choose_bins(np.abs(S), S>=0)))
        histbins = rdr.max() + 1
    assert all(np.all(np.diff(rdr[seg == s]) >= 0) for s in set(seg)) or testmethod != 'special'
    
    permute = generateperm(rep, seg, permutations, permethod)
    tesstat = generatestat(rdr, seg, rep, testmethod, meanmethod, histbins, maxrtrdr, quantile)
    if testmethod != 'special':
        obsstat = tesstat(rep)
        nulldis = np.fromiter((tesstat(perm_rep) for perm_rep in permute), float)
        pval_emp = np.sum(obsstat <= nulldis) / permutations
        a, b, loc, scale, p_of_fit, pval_fit, pval_comb = fit_and_get_pval(nulldis, obsstat, pval_emp)
    else:
        obsstat_early, obsstat_late = tesstat(rep)
        nulldis_early, nulldis_late = map(lambda x : np.array(x), zip(*(tesstat(perm_rep) for perm_rep in permute)))
        pval_emp_early, pval_emp_late = (np.sum(obsstat_early <= nulldis_early) / permutations,
                                         np.sum(obsstat_late <= nulldis_late) / permutations)
        a_early, b_early, loc_early, scale_early, p_of_fit_early, pval_fit_early, pval_comb_early = fit_and_get_pval(nulldis_early, obsstat_early, pval_emp_early)
        a_late, b_late, loc_late, scale_late, p_of_fit_late, pval_fit_late, pval_comb_late = fit_and_get_pval(nulldis_late, obsstat_late, pval_emp_late)
        a, b, loc, scale, p_of_fit, pval_fit, pval_comb, pval_emp, obsstat = (a_early, b_early, loc_early, scale_early, p_of_fit_early, pval_fit_early, pval_comb_early, pval_emp_early, obsstat_early) \
                                                                             if pval_comb_early <= pval_comb_late else \
                                                                             (a_late, b_late, loc_late, scale_late, p_of_fit_late, pval_fit_late, pval_comb_late, pval_emp_late, obsstat_late)

    sizes = celldf.groupby(seg)[timing].apply(lambda x : x[x.isin(['early', 'late'])].value_counts().min())
    means = celldf.groupby(seg)[[timing, rdr_profile]].apply(lambda x : np.mean(
                                                                        np.fromiter((x[rdr_profile][x[timing]=='early'].sample(sizes[x.name], replace=True).mean() -
                                                                                     x[rdr_profile][x[timing]=='late'].sample(sizes[x.name], replace=True).mean() 
                                                                                    for _ in range(100)),
                                                                                    dtype=float)))
    means = (means * sizes).sum() / sizes.sum()

    if toplot:
        plot_infer(celldf, nulldis_early, obsstat_early, a, b, loc, scale, p_of_fit, timing, rdr_profile)
    
    result = {'CELL' : cell,
              'S_TEST_STAT' : obsstat,
              'PVAL_EMP' : pval_emp,
              'PVAL_FIT' : pval_fit,
              'P_OF_FIT' : p_of_fit,
              'PVAL_COMB' : pval_comb,
              'MEAN_DIFF' : means,
              'IS_MAGENTA_HIGHER' : means >= 0}
    if testmethod == 'special':
        result.update({'PVAL_EMP_EARLY' : pval_emp_early,
                       'PVAL_FIT_EARLY' : pval_fit_early,
                       'P_OF_FIT_EARLY' : p_of_fit_early,
                       'PVAL_COMB_EARLY' : pval_comb_early,
                       'PVAL_EMP_LATE' : pval_emp_late,
                       'PVAL_FIT_LATE' : pval_fit_late,
                       'P_OF_FIT_LATE' : p_of_fit_late,
                       'PVAL_COMB_LATE' : pval_comb_late})
    return result


def generateperm(_rep, _spl, _permutations, _method, _minperm=1):
    assert _method in ['permutation' , 'roll']
    permute = np.random.permutation if 'permutation' else (lambda array : np.roll(array, np.random.randint(_minperm, len(array))))
    parts = tuple(_rep[l:r] for l, r in get_consecutive(np.unique(_spl, return_index=True)[1]))
    gcpermu = (lambda : np.concatenate([permute(part) for part in parts]))
    return (gcpermu() for _ in range(_permutations))


def generatestat(_rdr, _seg, _rep, _method, _meanmethod, _bins, _maxrdr, _quantile):
    parts = tuple(get_consecutive(np.unique(_seg, return_index=True)[1]))
    split1 = (lambda array : (array[l:r] for l, r in parts))
    split2 = (lambda array1, array2 : ((array1[l:r], array2[l:r]) for l, r in parts))
    gridsize = max(min(grep.sum(), (~grep).sum()) for grep in split1(_rep))
    buffer = max(rtkde(_rdr[_rep], gridsize, vmax=_maxrdr)[2], rtkde(_rdr[~_rep], gridsize, vmax=_maxrdr)[2]) if _method == 'kdensity' else None
    sumstat = choose_sumstat(_method, _bins, _maxrdr, gridsize, buffer, _quantile)
    if _meanmethod == 'weighted_min':
        weights = np.fromiter((min(grep.sum(), (~grep).sum()) for grep in split1(_rep)), float)
    elif _meanmethod == 'weighted_sum':
        weights = np.fromiter((len(grep) for grep in split1(_rep)), float)    
    elif _meanmethod == 'normal':
        weights = np.fromiter((1 for _ in split1(_rep)), float)
    else:
        assert False, 'Unknown mean method'
    tot = weights.sum()
    tonum = (lambda array : np.nan_to_num(array, nan=0, posinf=0, neginf=0))
    wmedian = (lambda X : np.median(np.concatenate([[x]*int(w) for x, w in zip(X, weights)])))
    if _method == 'special':
        hmean = (lambda E, L : (scipy.stats.hmean(np.array(E)[~np.isclose(E, 0)], weights=weights[~np.isclose(E, 0)]), 
                                scipy.stats.hmean(np.array(L)[~np.isclose(L, 0)], weights=weights[~np.isclose(L, 0)])))
        return (lambda rep : hmean(*zip(*(tonum(sumstat(grdr[grep], grdr[~grep])) for grdr, grep in split2(_rdr, rep)))))
    else:
        hmean = (lambda E : scipy.stats.hmean(E[~np.isclose(E, 0)], weights=weights[~np.isclose(E, 0)]))
        return (lambda rep : hmean(np.fromiter((tonum(sumstat(grdr[grep], grdr[~grep])) for grdr, grep in split2(_rdr, rep)), float)))
    

def choose_sumstat(_method, _bins, _maxrdr, _gridsize, _buffer, _quantile):
    if _method == 'special':
        sumstat = (lambda late, early : (max(1., (early > late[floor(late.size * (1.0 - _quantile))]).sum()) / early.size, 
                                         max(1., (late < early[min(ceil(early.size * _quantile), early.size - 1)]).sum()) / late.size))
    elif _method == 'density':
        hist = (lambda a : np.bincount(a, weights=np.full(a.size, 1./a.size), minlength=_bins+1))
        sumstat = (lambda late, early : 1 - np.sum(np.minimum(hist(late), hist(early))))
    elif _method == 'integral':
        hist = (lambda a : np.bincount(a, weights=np.full(a.size, 1./a.size), minlength=_bins+1))
        sumstat = (lambda late, early : 1 - scipy.integrate.simpson(np.minimum(hist(late), hist(early))))
    elif _method == 'kdensity':
        integrate = (lambda rtkde1, rtkde2 : scipy.integrate.simpson(np.minimum(rtkde1[0], rtkde2[0]), rtkde1[1]))
        sumstat = (lambda late, early : 1 - integrate(rtkde(late, _gridsize, vmax=_maxrdr, buffer=_buffer),
                                                      rtkde(early, _gridsize, vmax=_maxrdr, buffer=_buffer)))
    elif _method == 'earthmover':
        sumstat = (lambda late, early : scipy.stats.wasserstein_distance(early, late))
    elif _method == 'energy':
        sumstat = (lambda late, early : scipy.stats.energy_distance(early, late))
    elif _method == 'overlap':
        hist = (lambda a : np.bincount(a, minlength=_bins+1))
        sumstat = (lambda late, early : 1 - (np.sum(np.minimum(hist(early), hist(late)) / min(early.size, late.size))))
    elif _method == 'mean':
        sumstat = (lambda late, early : np.mean(early) - np.mean(late))
    elif _method == 'esmean':
        sumstat = (lambda late, early : (np.mean(early) - np.mean(late)) /
                                         np.sqrt(((len(early) - 1) * (np.std(early, ddof=1) ** 2) + 
                                                  (len(late) - 1) * (np.std(late, ddof=1) ** 2)) / (len(early) + len(late) - 2)))
    elif _method == 'median':
        sumstat = (lambda late, early : np.median(early) - np.median(late))
    elif _method == 'esmedian':
        sumstat = (lambda late, early : (np.median(early) - np.median(late)) /
                                         np.sqrt(((len(early) - 1) * (np.std(early, ddof=1) ** 2) + 
                                                  (len(late) - 1) * (np.std(late, ddof=1) ** 2)) / (len(early) + len(late) - 2)))
    elif _method == 'ks':
        step = np.linspace(0, 2, 100)
        sumstat = (lambda late, early : np.max((ECDF(late)(step) - ECDF(early)(step)) if late.size > 0 and early.size > 0 else 0))
    elif _method == 'mw':
        sumstat = (lambda late, early : scipy.stats.mannwhitneyu(early, late).statistic)
    elif _method == 'tt':
        sumstat = (lambda late, early : scipy.stats.ttest_ind(early, late, equal_var=True).statistic)
    else:
        assert False, 'Unknown teststat method'
    return sumstat


def fit_and_get_pval(nulldis, obsstat, pval_emp):
    a1, b1, loc1, scale1 = (None, None, None, None)
    a2, b2, loc2, scale2 = (None, None, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a1, b1, loc1, scale1 = scipy.stats.beta.fit(nulldis, method='MLE')
        a2, b2, loc2, scale2 = scipy.stats.beta.fit(nulldis, method='MM')
    _, p_of_fit1 = scipy.stats.kstest(nulldis, 'beta', args=(a1, b1, loc1, scale1), alternative='two-sided')
    _, p_of_fit2 = scipy.stats.kstest(nulldis, 'beta', args=(a2, b2, loc2, scale2), alternative='two-sided')
    a, b, loc, scale, p_of_fit = (a1, b1, loc1, scale1, p_of_fit1) if p_of_fit1 >= p_of_fit2 else (a2, b2, loc2, scale2, p_of_fit2)
    pval_fit = 1 - scipy.stats.beta.cdf(obsstat, a=a, b=b, loc=loc, scale=scale)
    pval_comb = pval_fit if p_of_fit > 0.05 \
                        else (pval_fit if ((pval_fit >= (pval_emp * 0.9)) & (pval_fit <= (pval_emp * 1.1)) | (pval_emp == 0.) & (pval_fit <= 0.01)) 
                        else pval_emp)
    return a, b, loc, scale, p_of_fit, pval_fit, pval_comb


def rtkde(data, gridsize, vmin=0., vmax=None, bw=None, cut=3, buffer=None):
    bw = bandwidths.select_bandwidth(data, "normal_reference", kernels.Gaussian()) if bw is None else bw
    buffer = bw if buffer is None else buffer
    vmax = np.max(data) if vmax is None else vmax
    gridsize = 2 ** np.ceil(np.log2(gridsize))
    L, R = (vmin - cut * buffer, vmax + cut * buffer)
    grid, d = np.linspace(L, R, int(gridsize), retstep=True)
    binned = fast_linbin(data, L, R, gridsize) / (d * len(data))
    z = silverman_transform(bw, gridsize, R - L) * forrt(binned)
    return revrt(z), grid, bw


def plot_infer(celldf, nulldis, obsstat, a, b, loc, scale, p_of_fit, timing, rdr_profile):
    celldf = celldf.sort_values(['CHR', 'START'])

    plt.figure()
    sns.histplot(nulldis, stat='density')
    plt.axvline(obsstat, color='red', ls='--', lw=2.5)
    plt.xlabel('Extent of non-overlap')
    plt.ylabel('Count')
    points = np.linspace(nulldis.min(), nulldis.max(), 100)
    plt.plot(points, scipy.stats.beta.pdf(points, a=a, b=b, loc=loc, scale=scale), c='black')
    plt.title(p_of_fit)

    plt.figure()
    sns.scatterplot(data=celldf, x='GC', y=rdr_profile, hue='consistent_rt', hue_order = ['late', 'early'], palette = ['green', 'magenta'], 
            s=10, edgecolor=None, linewidth=0, alpha=0.6, legend=True)
    plt.xlabel('GC content')
    plt.ylabel('RDR')
    df = celldf.reset_index(drop=True).sort_values('GC')
    def linear_regression(data, val='RDR'):
        slope, intercept = scipy.stats.linregress(data['GC'], data[val])[:2]
        return data['GC'] * slope + intercept, slope
    all_curve, all_param = linear_regression(df, val=rdr_profile)
    plt.plot(df['GC'], all_curve, ls='--', lw=2, c='k')
    plt.ylim(0, 3)
    plt.xlim(0.2, 0.7)

    plt.figure()
    sns.stripplot(data=celldf, x=timing, y=rdr_profile, hue=timing, hue_order = ['late', 'early'], palette = ['green', 'magenta'], s=3)

    plt.figure()
    sns.displot(data=celldf, x=rdr_profile, hue=timing, hue_order = ['late', 'early'], palette = ['green', 'magenta'])
    plt.xlabel('RDR')
    plt.ylabel('Count')

    plt.figure(figsize=(20, 4))
    sns.scatterplot(data=celldf, x='GENOME', y=rdr_profile, hue=timing, hue_order = ['late', 'early'], 
                    palette = ['green', 'magenta'], s=10, edgecolor=None, linewidth=0, alpha=0.6)
    xticks = celldf.groupby('CHR')['GENOME'].first()
    plt.plot((xticks.values, xticks.values), (np.full_like(xticks, 0.0), np.full_like(xticks, 3.0)), '--b', linewidth=1.0)
    plt.xticks(xticks.values, xticks.index, rotation=30)
    plt.ylim(0, 3)

