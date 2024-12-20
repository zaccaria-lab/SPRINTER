from .utils import *

from sklearn.neighbors import LocalOutlierFactor
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.cluster import hierarchy
import statsmodels.formula.api as smf



def infer_G2(total_counts, annotations, clones_all, cn_all, normal_clones, jobs=1, devmode=False, toplot=True, reps=20):
    clus = clones_all[~pd.isnull(clones_all['CLONE'])][['CELL', 'CLONE', 'IS_REASSIGNED']].reset_index(drop=True)
    clus['IS-S-PHASE'] = clus['CELL'].map(annotations.set_index('CELL')['IS-S-PHASE'])
    countsg1g2, countssphase = infer_ploidy_clones(clus, cn_all, total_counts, normal_clones, clones_all, annotations)
    inferG2 = find_G2(countsg1g2, countssphase, clus, jobs=jobs, devmode=devmode)
    annotations = annotations.merge(inferG2[['CELL', 'NONORM_PROB_G1', 'NONORM_PROB_G2', 'PROB_G1', 'PROB_G2', 'G1_COUNT_PVAL', 'G2_COUNT_PVAL', 'IS_G2']], on='CELL', how='left')
    annotations['IS_REASSIGNED'] = annotations['CELL'].map(clones_all.set_index('CELL')['IS_REASSIGNED'])
    annotations['TOTAL_COUNT'] = annotations['CELL'].map(total_counts)
    annotations['BASELINE_PLOIDY'] = annotations['CELL'].map(cn_all.groupby('CELL')['CN_TOT'].apply(lambda x : est_ploidy(x, reps)))
    with pd.option_context("future.no_silent_downcasting", True):
        annotations['IS_G2'] = annotations['IS_G2'].fillna(False).infer_objects(copy=False)
    annotations['PREDICTED_PHASE'] = np.where(annotations['IS-S-PHASE'], 'S', np.where(annotations['IS_G2'], 'G2', 'G1'))
    annotations['PREDICTED_CLONE'] = annotations['CELL'].map(clones_all[['CELL', 'CLONE']].drop_duplicates().set_index('CELL')['CLONE'])
    if toplot: 
        plot_final_heatmap(cn_all, clones_all, annotations, clus)
    return annotations


est_ploidy = (lambda D, reps : np.mean([D.sample(len(D), replace=True).mean() for _ in range(reps)]))


def infer_ploidy_clones(clus, cn_all, total_counts, normal_clones, clones_all, annotations, reps=20):
    selected = clus[~clus['IS-S-PHASE']][['CELL', 'CLONE']].drop_duplicates().set_index('CELL')['CLONE']
    data = cn_all[cn_all['CELL'].isin(selected.index)].reset_index(drop=True)
    data['CLONE'] = data['CELL'].map(selected)
    data['IS_REASSIGNED'] = data['CELL'].map(clus.set_index('CELL')['IS_REASSIGNED'])
    except_plclones = clus.groupby('CLONE')['IS_REASSIGNED'].apply(lambda x : (~x).sum())
    except_plclones = except_plclones[except_plclones == 0].index.to_numpy()
    ploidies = data[(~data['IS_REASSIGNED']) | (data['CLONE'].isin(except_plclones))].groupby(['CLONE', 'CHR', 'START', 'END'])['CN_TOT'].apply(lambda x : x.value_counts().index[0]).reset_index().groupby('CLONE')['CN_TOT'].apply(lambda x : est_ploidy(x, reps))
    ploidies = ploidies.rename('PLOIDY').sort_values()
    ploidies_diff = ((np.maximum(ploidies, ploidies.shift(1)) / np.minimum(ploidies, ploidies.shift(1))).fillna(np.inf) > 1.2)
    ploidy_clones = (ploidies_diff | ploidies_diff.index.to_series().isin(normal_clones) | ploidies_diff.index.to_series().shift(1).isin(normal_clones)).astype(int).cumsum().rename('PLOIDY_CLONE')
    data['PLOIDY_CLONE'] = data['CLONE'].map(ploidy_clones)
    clus['PLOIDY_CLONE'] = clus['CLONE'].map(ploidy_clones)
    data = data[['CELL', 'CLONE', 'PLOIDY_CLONE', 'IS_REASSIGNED']].drop_duplicates().reset_index(drop=True)
    assert not pd.isnull(data['PLOIDY_CLONE']).any()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data['RAW_TOTAL_COUNT'] = data['CELL'].map(total_counts)
        data['PLOIDY_CELL'] = data['CELL'].map(cn_all.groupby('CELL')['CN_TOT'].mean())
        model_reads = max((smf.quantreg('RAW_TOTAL_COUNT ~ np.exp(-PLOIDY_CELL)', data[~data['IS_REASSIGNED']]).fit(q=.5),
                        smf.quantreg('RAW_TOTAL_COUNT ~ np.log(PLOIDY_CELL)', data[~data['IS_REASSIGNED']]).fit(q=.5)),
                        key=(lambda x : x.prsquared))
        data['READS_CORR'] = model_reads.predict(data[['PLOIDY_CELL']])
        data['TOTAL_COUNT'] = ((data['RAW_TOTAL_COUNT'] / data['READS_CORR']) * data['RAW_TOTAL_COUNT'].mean()).round().astype(int)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data, x='PLOIDY_CELL', y='RAW_TOTAL_COUNT', hue='IS_REASSIGNED')
    xs = np.linspace(data['PLOIDY_CELL'].min(), data['PLOIDY_CELL'].max(), 1000)
    plt.plot(xs, model_reads.predict(pd.DataFrame({'PLOIDY_CELL' : xs})), color='red')
    plt.savefig('reads_ploidy.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=data, x='PLOIDY_CELL', y='TOTAL_COUNT', hue='IS_REASSIGNED')
    plt.savefig('corrected_reads.png', dpi=300, bbox_inches='tight')

    assert not pd.isnull(data['TOTAL_COUNT']).any()

    countssphase = total_counts[annotations[annotations['IS-S-PHASE']==True]['CELL'].unique()].reset_index()
    countssphase['CLONE'] = countssphase['CELL'].map(clones_all.set_index('CELL')['CLONE'])
    countssphase['PLOIDY_CLONE'] = countssphase['CLONE'].map(ploidy_clones)
    countssphase = countssphase.dropna().reset_index(drop=True)
    return data, countssphase


def find_G2(counts, countssphase, clus, jobs=1, devmode=False, group='CLONE'):
    inferG2 = None
    sthres = counts[['CELL', 'CLONE', 'PLOIDY_CLONE', 'TOTAL_COUNT']].reset_index(drop=True)
    sthres['GROUP_STHRES'] = sthres[group].map(countssphase.groupby(group)['COUNT'].mean())
    sthres['STHRES'] = sthres['TOTAL_COUNT'] >= sthres['GROUP_STHRES']
    sthres['STHRES'] = sthres.groupby(group)['STHRES'].transform(lambda x : x.sum() / len(x))
    sthres['STHRES'] = sthres.groupby('PLOIDY_CLONE')['STHRES'].transform('mean')
    sthres = sthres.groupby(group)['STHRES'].first().to_dict()
    with Manager() as manager:
        shared = manager.list()
        with Pool(processes=min(jobs, counts[group].nunique()), 
                  initializer=init_infer_g2_perploidy, 
                  initargs=(manager.dict({ploclone : df for ploclone, df in counts.groupby(group)}),
                            manager.dict({ploclone : df for ploclone, df in clus.groupby(group)}),
                            manager.dict(sthres),
                            shared,
                            devmode)) as pool:
            bar = ProgressBar(total=counts[group].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Ploidy clone {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [clone for clone in pool.imap_unordered(infer_g2_perploidy, counts[group].unique()) if progress(clone)]
        inferG2 = pd.concat(shared)
    return inferG2


def init_infer_g2_perploidy(_COUNTS, _CLUS, _STHRES, _SHARED, _DEVMODE):
    global COUNTS, CLUS, STHRES, SHARED, DEVMODE
    COUNTS = _COUNTS
    CLUS = _CLUS
    STHRES = _STHRES
    SHARED = _SHARED
    DEVMODE = _DEVMODE


def infer_g2_perploidy(clone, nbinom=True, soft=True, max_frac=.5, minmax_frac=.05):
    num_all = len(CLUS[clone])
    num_sphase = len(CLUS[clone][CLUS[clone]['IS-S-PHASE']==True])
    counts = COUNTS[clone][['CELL', 'CLONE', 'IS_REASSIGNED', 'PLOIDY_CLONE', 'TOTAL_COUNT']].reset_index(drop=True)
    assert num_all == (num_sphase + len(counts))
    assert set(counts['CELL'].unique()) == set(CLUS[clone][CLUS[clone]['IS-S-PHASE']==False]['CELL'].unique())
    counts_tofit = counts['TOTAL_COUNT'].values
    if len(counts_tofit) > 10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = LocalOutlierFactor(n_jobs=1).fit_predict(counts_tofit.reshape(-1, 1)) == 1
            if DEVMODE:
                pd.DataFrame({'COUNTS' : counts_tofit, 'OUT' : out}).to_csv('clone{}_counts_prefilter.tsv.gz'.format(clone), sep='\t')
            counts_tofit = counts_tofit[out]

    maxfrac_sthres = STHRES[clone] if clone in STHRES else 0.
    if maxfrac_sthres < minmax_frac:
        maxfrac_sthres = min(num_sphase / len(counts), max_frac)
    maxfrac_sthres = max(maxfrac_sthres, minmax_frac)
    if DEVMODE:
        plt.figure()
        sns.histplot(counts_tofit)
        plt.axvline(np.quantile(counts_tofit, 1. - maxfrac_sthres), c='red', ls='--', lw=3)
        plt.savefig('clone{}_counts_tofit.png'.format(clone), dpi=300, bbox_inches='tight')
        plt.title('Max G2 frac informed from S phase: {}'.format(maxfrac_sthres))
        pd.Series(counts_tofit).rename('TOTAL_COUNT').to_csv('clone{}_counts_tofit.tsv.gz'.format(clone), sep='\t')
    if len(counts_tofit) > 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fracg2, exp_sfrac = importance_sampling_frac_g2(counts_tofit, np.clip(num_sphase / len(counts), .0, 1.),
                                                            max_frac=maxfrac_sthres, nbinom=nbinom, soft=soft, name=clone, devmode=DEVMODE)[:2]
            pdfg1, pdfg2, cdfg1, cdfg2 = llk_frac_g2(counts_tofit, fracg2, exp_sfrac=exp_sfrac, nbinom=nbinom, soft=soft)[2:]
        if all(r is not None for r in [pdfg1, pdfg2, cdfg1, cdfg2]):
            compg1 = pdfg1(counts['TOTAL_COUNT'])
            compg2 = pdfg2(counts['TOTAL_COUNT'])
            counts['NONORM_PROB_G1'] = compg1
            counts['NONORM_PROB_G2'] = compg2
            counts['PROB_G1'] = compg1 / (compg1 + compg2)
            counts['PROB_G2'] = compg2 / (compg1 + compg2)
            counts['G1_COUNT_PVAL'] = 1 - cdfg1(counts['TOTAL_COUNT'])
            counts['G2_COUNT_PVAL'] = cdfg2(counts['TOTAL_COUNT'])
            counts['IS_G2'] = counts['PROB_G1'] < .3
        else:
            counts['NONORM_PROB_G1'] = np.nan
            counts['NONORM_PROB_G2'] = np.nan
            counts['PROB_G1'] = 1.
            counts['PROB_G2'] = 0.
            counts['G1_COUNT_PVAL'] = 1.
            counts['G2_COUNT_PVAL'] = 0.
            counts['IS_G2'] = (counts['TOTAL_COUNT'] >= np.quantile(counts_tofit, 1 - fracg2, method='closest_observation'))
    else:
        pdfg1, pdfg2 = (None, None)
        counts['NONORM_PROB_G1'] = np.nan
        counts['NONORM_PROB_G2'] = np.nan
        counts['PROB_G1'] = 1.
        counts['PROB_G2'] = 0.
        counts['G1_COUNT_PVAL'] = 1.
        counts['G2_COUNT_PVAL'] = 0.
        counts['IS_G2'] = False
    plt.figure()
    xs = np.arange(counts['TOTAL_COUNT'].min(), counts['TOTAL_COUNT'].max() + 1)
    if counts['IS_G2'].nunique() == 2:
        sns.histplot(data=counts, x='TOTAL_COUNT', hue='IS_G2', palette=['green', 'red'], stat='density', alpha=.3)
    else:
        sns.histplot(data=counts, x='TOTAL_COUNT', hue='IS_G2', stat='density', alpha=.3)
    if pdfg1 is not None and pdfg2 is not None and all(r is not None for r in [pdfg1, pdfg2, cdfg1, cdfg2]):
        plt.plot(xs, pdfg1(xs), '--g', lw=3)
        plt.plot(xs, pdfg2(xs), '--r', lw=3)
    plt.savefig('clone{}_fitted_counts.png'.format(clone), dpi=300, bbox_inches='tight')
    SHARED.append(counts)
    return '{} ({} cells)'.format(clone, len(counts_tofit))


def importance_sampling_frac_g2(counts, _exp_sfrac, nsamples=2000, scale_above=.1,
                                max_frac=.5, min_frac=.02, nbinom=True, soft=True, inv_nsamples=int(1e5), name='', devmode=False):
    max_frac = max(min_frac*2, max_frac)
    exp_sfrac = min(_exp_sfrac, max_frac)
    above_s = int((max_frac - exp_sfrac) / (max_frac - min_frac) * scale_above * nsamples)
    below_s = nsamples - above_s if exp_sfrac > min_frac else 0
    fracs = np.sort(np.concatenate((np.linspace(min_frac, exp_sfrac, below_s), np.linspace(exp_sfrac, max_frac, above_s))))
    logliks = np.array([llk_frac_g2(counts, frac, exp_sfrac, nbinom=nbinom, soft=soft)[0] for frac in fracs])
    fitted = ~np.isnan(logliks)
    if (fitted.sum() / len(fitted)) > 0.2 or not nbinom:
        fracs = fracs[fitted]
        logliks = logliks[fitted]
    else:
        logliks = np.array([llk_frac_g2(counts, frac, exp_sfrac, nbinom=False, soft=soft)[0] for frac in fracs])
        fitted = ~np.isnan(logliks)
        fracs = fracs[fitted]
        logliks = logliks[fitted]
    assert len(fracs) > 0 and len(logliks) > 0
    if devmode:
        plt.figure()
        plt.plot(fracs, logliks)
        plt.savefig('clone{}_importance_weights.png'.format(name), dpi=300, bbox_inches='tight')
    weights = (logliks + np.abs(np.min(logliks))) if (logliks < 0).sum() > 0 else logliks
    cdf = (weights / weights.sum()).cumsum()
    inv_cdf = scipy.interpolate.interp1d(cdf, fracs)
    inv_samples = np.linspace(cdf.min(), cdf.max(), inv_nsamples)
    post_sample = inv_cdf(inv_samples)
    xs = np.linspace(post_sample.min(), post_sample.max(), 100)
    post_kde = scipy.stats.gaussian_kde(post_sample)
    post_pdf = post_kde.pdf(xs)
    if devmode:
        plt.figure()
        sns.histplot(post_sample, stat='density', alpha=.3, bins=100)
        plt.plot(xs, post_pdf, '--k', lw=3)
    hist, xs = np.histogram(post_sample, bins=100)
    post_max = xs[hist.argmax()]
    if devmode:
        plt.axvline(post_max, ls='--', lw=4, c='r')
        plt.savefig('clone{}_posterior_fracg2.png'.format(name), dpi=300, bbox_inches='tight')
    return post_max, exp_sfrac, post_sample, post_kde


def llk_frac_g2(counts, frac, exp_sfrac, nbinom=True, soft=True):
    fit = scipy.stats.t.fit if not nbinom else (lambda D : convert_nbinom_params(NegativeBinomial(exog=np.ones(len(D)), endog=D).fit(disp=False).params))
    lik = scipy.stats.t.pdf if not nbinom else scipy.stats.nbinom.pmf
    loglik = scipy.stats.t.logpdf if not nbinom else scipy.stats.nbinom.logpmf
    cdf = scipy.stats.t.cdf if not nbinom else scipy.stats.nbinom.cdf

    X = counts
    L, R = X[X <= np.quantile(X, 1 - frac, method='closest_observation')], X[X >= np.quantile(X, 1 - frac, method='closest_observation')]
    boot_mean = (lambda D : np.random.choice(D, size=(len(D), 200), replace=True).mean(axis=0).mean())

    modeL, modeR = (boot_mean(L), boot_mean(R))
    fitL, fitR = (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cntL = np.concatenate((L[L <= modeL], 2 * modeL - L[L <= modeL]))
        if len(np.unique(cntL)) == 1:
            cntL[0] += 1
        fitL = fit(cntL)
        
        cntR = np.concatenate((R[R >= modeR], 2 * modeR - R[R >= modeR]))
        if len(np.unique(cntR)) == 1:
            cntR[0] += 1

        fitR = fit(cntR)
        if nbinom:
            n = min(fitL[0], fitR[0])
            prev_mean = (fitR[0] - fitR[0] * fitR[1]) / fitR[1]
            p = n / (n + prev_mean)
            fitR = (n, p)
        else:
            fitR = (fitR[0], fitR[1], max(fitR[2], fitL[2]))
        
        if nbinom:
            check = (lambda X : np.any(np.isnan(X)) or 
                                np.any(np.isinf(X)) or 
                                np.any(np.isclose(X[0], 0., rtol=1e-03, atol=1e-03)) or
                                X[0] > 1e10 or
                                np.any(np.isclose(X[1], 1., rtol=1e-03, atol=1e-03)) or
                                X[1] < 1e-15)
            if check(np.array(fitL)) or check(np.array(fitR)):
                return [np.nan, None, None, None, None, None]
        if soft:
            loglikL = np.nan_to_num(loglik(X, *fitL)) + np.log(1 - exp_sfrac)
            loglikR = np.nan_to_num(loglik(X, *fitR)) + np.log(exp_sfrac)
            llk = np.logaddexp(loglikL, loglikR)

        else:
            llk = np.maximum(np.nan_to_num(loglik(X, *fitL)), np.nan_to_num(loglik(X, *fitR)))

        return (np.mean(np.nan_to_num(llk)),
                (X, L, R),
                (lambda D : lik(D, *fitL)),
                (lambda D : lik(D, *fitR)),
                (lambda D : cdf(D, *fitL)),
                (lambda D : cdf(D, *fitR)))


def convert_nbinom_params(params):
    mu = np.exp(params[0])
    p = np.clip(1 / ((1 + mu) * params[1]), 0., 1.)
    n = mu * p / (1 - p)
    return n, p


def plot_final_heatmap(cn_all, clones_all, annotations, clus):
    heatmap_df = cn_all.merge(annotations[['CELL', 'PREDICTED_PHASE']], on='CELL', how='inner')
    color_df = heatmap_df[['CELL', 'CLONE', 'PREDICTED_PHASE']].drop_duplicates()
    color_df['PHASE_ORDER'] = color_df['PREDICTED_PHASE'].map({'G1' : 0, 'S' : 1, 'G2' : 2})
    table = pd.pivot_table(heatmap_df, values='CN_TOT', index=['CELL'], columns=['CHR', 'START', 'END'])
    order_leaves = (lambda vals : np.argsort(hierarchy.leaves_list(hierarchy.linkage(vals, method='weighted', metric='hamming', optimal_ordering=True))))
    color_df['CLONE_ORDER'] = color_df[['CLONE', 'PREDICTED_PHASE', 'CELL']].fillna('None')\
                              .groupby(['CLONE', 'PREDICTED_PHASE'])['CELL'].transform(lambda cells : order_leaves(table.loc[cells].values)
                                                                                                      if len(cells) > 3 else
                                                                                                      np.arange(len(cells)))
    color_df = color_df.sort_values(['CLONE', 'PHASE_ORDER', 'CLONE_ORDER'])
    color_df['CLONE_COLOR'] = color_df['CLONE'].map(dict(zip(color_df['CLONE'].unique(), sns.color_palette('tab20', color_df['CLONE'].nunique()))))
    color_df['CLONE_COLOR'] = color_df[['CLONE', 'CLONE_COLOR']].apply(lambda row : (102/255, 37/255, 6/255) if np.isnan(row['CLONE']) else row['CLONE_COLOR'], axis=1)
    color_df['PHASE_COLOR'] = color_df['PREDICTED_PHASE'].map({'G1' : '#d9d9d9', 'S' : '#737373', 'G2' : '#000000'})
    color_df = color_df.merge(clus[['CELL', 'PLOIDY_CLONE']].drop_duplicates(), on='CELL', how='left')
    color_df['PLOIDY_CLONE_COLOUR'] = color_df['PLOIDY_CLONE'].map(dict(zip(color_df['PLOIDY_CLONE'].unique(), sns.color_palette('Dark2', color_df['PLOIDY_CLONE'].nunique()))))
    table = table.reindex(color_df['CELL'].unique(), axis=0)
    sns.clustermap(table, row_cluster=False, col_cluster=False, cmap='RdBu_r', center=2, vmin=1, vmax=6, xticklabels=False, yticklabels=False, rasterized=True,
                    row_colors=color_df.set_index('CELL')[['PLOIDY_CLONE_COLOUR', 'CLONE_COLOR', 'PHASE_COLOR']])
    handles = [mpatches.Patch(facecolor=color) for color in color_df['PHASE_COLOR'].unique()]
    plt.legend(handles, color_df['PREDICTED_PHASE'].unique(), title='Phase', loc='upper left', bbox_to_anchor=(2, 1))
    plt.savefig('sprinter_finalresult.png', dpi=300, bbox_inches='tight')

