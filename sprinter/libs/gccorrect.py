from utils import *
from rtsegment import globalhmm

from sklearn.mixture import GaussianMixture
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess



def estimate_gcbiases(_data, gccorr, jobs):
    gc_res = None
    with Manager() as manager:
        with Pool(processes=min(jobs, _data['CELL'].nunique()),
                  initializer=init_fit_gc,
                  initargs=(manager.dict({cell : celldf for cell, celldf in _data.groupby('CELL')}), gccorr)) \
        as pool:
            bar = ProgressBar(total=_data['CELL'].nunique(), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            gc_res = pd.DataFrame([r for res in pool.imap_unordered(fit_gc, _data['CELL'].unique()) if progress(res[0]['CELL']) for r in res])
    gcbias_all = correct_gcbias(gc_res[gc_res['TIME'] == 'ALL'].reset_index(drop=True).set_index('CELL'), label='ALL')
    gcbias_early = correct_gcbias(gc_res[gc_res['TIME'] == 'EARLY'].reset_index(drop=True).set_index('CELL'), label='EARLY')
    gcbias_late = correct_gcbias(gc_res[gc_res['TIME'] == 'LATE'].reset_index(drop=True).set_index('CELL'), label='LATE')
    return {'ALL' : gcbias_all.to_dict('index'),
            'EARLY' : gcbias_early.to_dict('index'),
            'LATE' : gcbias_late.to_dict('index')}, gc_res


def init_fit_gc(_data, _gccorr):
    global DATA, GCCORR
    DATA = _data
    GCCORR = _gccorr


def fit_gc(cell):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        celldf = DATA[cell].sort_values('GC')
        params_all = est_gc_params(celldf.sort_values('GC').reset_index(drop=True))
        params_early = est_gc_params(celldf[celldf['consistent_rt']=='early'].sort_values('GC').reset_index(drop=True))
        params_late = est_gc_params(celldf[celldf['consistent_rt']=='late'].sort_values('GC').reset_index(drop=True))
        return [{'CELL' : cell,
                'TIME' : 'ALL',
                'SLOPE' : params_all[0],
                'INTERCEPT' : params_all[1]},
                {'CELL' : cell,
                'TIME' : 'LATE',
                'SLOPE' : params_late[0],
                'INTERCEPT' : params_late[1]},
                {'CELL' : cell,
                'TIME' : 'EARLY',
                'SLOPE' : params_early[0],
                'INTERCEPT' : params_early[1]}]


def est_gc_params(celldf):
    if GCCORR in ['QUANTILE', 'TEST']:
        form = (lambda Q : (Q.params['GC'], Q.params['Intercept']))
        return form(smf.quantreg('RAW_RDR ~ GC', data=celldf).fit(q=.5))
    elif GCCORR in ['MODAL']:
        df = celldf.sort_values('GC').reset_index(drop=True)
        df['GC_CLASS'] = np.digitize(df['GC'], np.linspace(0, 1, 100))
        gc_class_size = df.groupby('GC_CLASS')['RAW_RDR'].nunique().to_dict()
        df = df[df['GC_CLASS'].isin([gc_class for gc_class, gc_class_value in gc_class_size.items() if gc_class_value > 3])].reset_index(drop=True)
        median_gc = df.groupby('GC_CLASS')['GC'].median().to_dict()
        kdes = df.groupby('GC_CLASS')['RAW_RDR'].apply(lambda rdrs : scipy.stats.gaussian_kde(rdrs.values)).to_dict()
        objective = (lambda _slope, _intercept : -sum(kdes[gc_class].logpdf(_slope*median_gc[gc_class] + _intercept)[0]*gc_class_size[gc_class] for gc_class in kdes.keys()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slopes, intercepts, objs = coordinate_descent(_obj=objective, _df=df)
        return (slopes, intercepts)
    else:
        raise ValueError("Found not existing gccorr method")


def correct_gcbias(linregress, label, min_cells=100):
    linregress['RAW_SLOPE'] = linregress['SLOPE']
    slopes = linregress['SLOPE'].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='diag', n_init=10).fit(slopes)
    linregress['PREDICT'] = gmm.predict(slopes)
    params = list(zip(gmm.means_[:,0], np.sqrt(gmm.covariances_.T[0])))
    out_predict = max(linregress['PREDICT'].unique(), key=(lambda x : scipy.stats.norm.interval(0.95, loc=params[x][0], scale=params[x][1])[1]))
    maj_predict = [p for p in linregress['PREDICT'].unique() if p != out_predict][0]
    linregress['IS_OUT'] = (linregress['PREDICT'] == out_predict) & (linregress['SLOPE'] > scipy.stats.norm.interval(0.9, *params[maj_predict])[1]) & (linregress['SLOPE'] >= 0.0)
    is_corrected = False
    if ((linregress['IS_OUT'].value_counts()[False] / len(linregress)) >= 0.5) and (linregress['IS_OUT'].any()) and (len(linregress) >= min_cells) \
        and (params[maj_predict][0] < 2):
        is_corrected = True
        linregress['SLOPE'] = linregress['SLOPE'].where(~linregress['IS_OUT'], params[maj_predict][0])
        ref_slopes = linregress[~linregress['IS_OUT']]['SLOPE']
        corr_intercepts = linregress['SLOPE'].apply(lambda slope : linregress['INTERCEPT'].loc[(ref_slopes - slope).abs().idxmin()])
        linregress['INTERCEPT'] = linregress['INTERCEPT'].where(~linregress['IS_OUT'], corr_intercepts)
    plt.figure(figsize=(8, 6))
    sns.histplot(data=linregress, x='RAW_SLOPE', hue='IS_OUT', bins=100, stat='density', color='lightgray', palette='bright')
    xs = np.linspace(linregress['RAW_SLOPE'].min(), linregress['RAW_SLOPE'].max(), 100)
    for x in [maj_predict, out_predict]:
        plt.plot(xs, scipy.stats.norm.pdf(xs, loc=params[x][0], scale=params[x][1]), '--', lw=4)
    plt.title('{} GC outliers corrected'.format(linregress['IS_OUT'].value_counts()[True]) if is_corrected else 'No GC outliers corrected')
    plt.savefig('gccorr_{}.png'.format(label), dpi=600, bbox_inches='tight')
    linregress.to_csv('gccorr_{}.tsv.gz'.format(label), sep='\t')
    return linregress


def quantile_gccorrect_rdr(_data, _gcbiases, nocorrgcintercept, strictgc, _timing, _maxrdr, toplot, boot_samples=10000, gc_pieces=100):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _data[['GC', 'RAW_RDR', _timing]].dropna().reset_index(drop=True).sort_values('GC', ascending=True)
        df['GC_CLASS'] = np.digitize(df['GC'], np.linspace(0, 1, gc_pieces))
        df['RDR_CLASS'] = np.digitize(df['RAW_RDR'], np.linspace(0, 3, 100))

        params = None
        if _gcbiases is not None:
            cell = _data['CELL'].unique()[0]
            params = [(_gcbiases['ALL'][cell]['SLOPE'], _gcbiases['ALL'][cell]['INTERCEPT'], 'ALL'),
                      (_gcbiases['EARLY'][cell]['SLOPE'], _gcbiases['EARLY'][cell]['INTERCEPT'], 'EARLY'), 
                      (_gcbiases['LATE'][cell]['SLOPE'], _gcbiases['LATE'][cell]['INTERCEPT'], 'LATE')]
        else:
            call = smf.quantreg('RAW_RDR ~ GC', data=df).fit(q=.5)
            early = smf.quantreg('RAW_RDR ~ GC', data=df[df[_timing]=='early']).fit(q=.5)
            late = smf.quantreg('RAW_RDR ~ GC', data=df[df[_timing]=='late']).fit(q=.5)
            params = [(D.params['GC'], D.params['Intercept'], T) for D, T in [(call, 'ALL'), (early, 'EARLY'), (late, 'LATE')]]

        if strictgc or any(p[0] > 2 for p in params):
            chosen = max(params, key=(lambda x : x[0]))[:2]
        else:
            chosen = (np.mean([p[0] for p in params]), np.mean([p[1] for p in params]))
        _corr = df['GC'] * chosen[0] + chosen[1]
        chosen_slope = chosen[0]
        chosen_intercept = chosen[1]

        medians = df.groupby('GC_CLASS')['RAW_RDR'].transform('median')
        opt = (lambda mus : (np.power(mus - medians, 2)).sum())
        h = scipy.optimize.minimize_scalar((lambda h : opt(_corr + h)), method='brent').x

        if toplot:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=df, x='GC', y='RAW_RDR', hue='consistent_rt', hue_order=['not_consistent', 'late', 'early'], palette=['lightgray', 'green', 'magenta'], s=4, alpha=.5, legend=False)
            plt.xlim(.2, .7)
            plt.ylim(0, 2.2)
            plt.plot(df['GC'], call.predict(df[['GC']]), ls='--', lw=2, c='black')
            plt.plot(df['GC'], late.predict(df[['GC']]), ls='--', lw=2, c='green')
            plt.plot(df['GC'], early.predict(df[['GC']]), ls='--', lw=2, c='magenta')

        correction = ((df['consistent_rt'] == 'late').astype(int) * (_corr + h) + \
                      (df['consistent_rt'] == 'early').astype(int) * (_corr + h) + \
                      (df['consistent_rt'] == 'not_consistent').astype(int) * _corr)
        return correction, {'GC_SLOPE' : chosen_slope,
                            'GC_INTERCEPT' : chosen_intercept + h}


def modal_gccorrect_rdr(_data, _gcbiases, nocorrgcintercept, strictgc, _timing, _maxrdr, toplot, boot_samples=10000, gc_pieces=100):    
    df = _data[['GC', 'RAW_RDR', _timing]].dropna().reset_index(drop=True).sort_values('GC', ascending=True)
    df['GC_CLASS'] = np.digitize(df['GC'], np.linspace(0, 1, gc_pieces))
    df_early = df[df[_timing] == 'early'].reset_index(drop=True)
    df_late = df[df[_timing] == 'late'].reset_index(drop=True)

    gc_class_size = {'all' : df.groupby('GC_CLASS')['RAW_RDR'].nunique().to_dict(),
                    'early' : df_early.groupby('GC_CLASS')['RAW_RDR'].nunique().to_dict(),
                    'late' : df_late.groupby('GC_CLASS')['RAW_RDR'].nunique().to_dict()}

    df_all = df[df['GC_CLASS'].isin([gc_class for gc_class, gc_class_value in gc_class_size['all'].items() if gc_class_value > 3])].reset_index(drop=True)
    df_early = df_early[df_early['GC_CLASS'].isin([gc_class for gc_class, gc_class_value in gc_class_size['early'].items() if gc_class_value > 3])].reset_index(drop=True)
    df_late = df_late[df_late['GC_CLASS'].isin([gc_class for gc_class, gc_class_value in gc_class_size['late'].items() if gc_class_value > 3])].reset_index(drop=True)

    median_gc = {'all' : df_all.groupby('GC_CLASS')['GC'].median().to_dict(),
                'early' : df_early.groupby('GC_CLASS')['GC'].median().to_dict(),
                'late' : df_late.groupby('GC_CLASS')['GC'].median().to_dict()}

    kdes = {'all' :  df_all.groupby('GC_CLASS')['RAW_RDR'].apply(lambda rdrs : scipy.stats.gaussian_kde(rdrs.values)).to_dict(),
            'early' : df_early.groupby('GC_CLASS')['RAW_RDR'].apply(lambda rdrs : scipy.stats.gaussian_kde(rdrs.values)).to_dict(),
            'late' :  df_late.groupby('GC_CLASS')['RAW_RDR'].apply(lambda rdrs : scipy.stats.gaussian_kde(rdrs.values)).to_dict()}

    objective = { 'all' : (lambda _slope, _intercept : -sum(kdes['all'][gc_class].logpdf(_slope*median_gc['all'][gc_class] + _intercept)[0]*gc_class_size['all'][gc_class] for gc_class in kdes['all'].keys())),
                'early' : (lambda _slope, _intercept : -sum(kdes['early'][gc_class].logpdf(_slope*median_gc['early'][gc_class] + _intercept)[0]*gc_class_size['early'][gc_class] for gc_class in kdes['early'].keys())),
                'late' : (lambda _slope, _intercept : -sum(kdes['late'][gc_class].logpdf(_slope*median_gc['late'][gc_class] + _intercept)[0]*gc_class_size['late'][gc_class] for gc_class in kdes['late'].keys())) }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = None
        if _gcbiases is not None:
            cell = _data['CELL'].unique()[0]
            params = [(_gcbiases['ALL'][cell]['SLOPE'], _gcbiases['ALL'][cell]['INTERCEPT'], 'ALL'),
                      (_gcbiases['EARLY'][cell]['SLOPE'], _gcbiases['EARLY'][cell]['INTERCEPT'], 'EARLY'), 
                      (_gcbiases['LATE'][cell]['SLOPE'], _gcbiases['LATE'][cell]['INTERCEPT'], 'LATE')]
        else:
            all_slope, all_intercept, all_objective = coordinate_descent(_obj=objective['all'], _df=df_all)
            early_slope, early_intercept, early_objective = coordinate_descent(_obj=objective['early'], _df=df_early)
            late_slope, late_intercept, late_objective = coordinate_descent(_obj=objective['late'], _df=df_late)
            params = [(all_slope, all_intercept, 'ALL'),
                      (early_slope, early_intercept, 'EARLY'),
                      (late_slope, late_intercept, 'LATE')]

        chosen = (np.mean([p[0] for p in params]), np.mean([p[1] for p in params]))
        _corr = df['GC'] * chosen[0] + chosen[1]
        chosen_slope = chosen[0]
        chosen_intercept = chosen[1]
        minimize = scipy.optimize.minimize_scalar
        h = minimize((lambda _h : objective['all'](chosen_slope, chosen_intercept + _h)), method='brent').x
    
        if toplot:
            print('h {}'.format(h))
            print('all_slope {}'.format(all_slope), 'all_obj {}'.format(all_objective), 
                  '\nearly_slope {}'.format(early_slope), 'early_obj {}'.format(early_objective), 
                  '\nlate_slope {}'.format(late_slope), 'late_obj {}'.format(late_objective))            
            kdes_maxima = {'all' : [(median_gc['all'][gc_class], (scipy.optimize.minimize_scalar(lambda x: -kdes['all'][gc_class].pdf(x))).x[0]) for gc_class in kdes['all'].keys()] ,
                           'early' : [(median_gc['early'][gc_class], (scipy.optimize.minimize_scalar(lambda x: -kdes['early'][gc_class].pdf(x))).x[0]) for gc_class in kdes['early'].keys()],
                           'late' : [(median_gc['late'][gc_class], (scipy.optimize.minimize_scalar(lambda x: -kdes['late'][gc_class].pdf(x))).x[0]) for gc_class in kdes['late'].keys()]}
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x='GC', y='RAW_RDR', hue='consistent_rt', hue_order=['not_consistent', 'late', 'early'], palette=['lightgray', 'green', 'magenta'], s=2, alpha=0.7, legend=False)
            plt.xlim(.2, .7)
            plt.ylim(0, 3)
            for pair in kdes_maxima['early']:
                plt.plot(pair[0], pair[1], marker='.', ms=8, c='magenta')
            for pair in kdes_maxima['late']:
                plt.plot(pair[0], pair[1], marker='.', ms=8, c='darkgreen')
            for pair in kdes_maxima['all']:
                plt.plot(pair[0], pair[1], marker='.', ms=8, c='black')
            plt.plot(df['GC'], (all_slope * df['GC'] + all_intercept), ls='--', lw=2, c='black')
            plt.plot(df['GC'], (late_slope * df['GC'] + late_intercept), ls='--', lw=2, c='green')
            plt.plot(df['GC'], (early_slope * df['GC'] + early_intercept), ls='--', lw=2, c='magenta')

        correction = ((df['consistent_rt'] == 'late').astype(int) * (_corr + h) + \
                      (df['consistent_rt'] == 'early').astype(int) * (_corr + h) + \
                      (df['consistent_rt'] == 'not_consistent').astype(int) * _corr)
        return correction, {'GC_SLOPE' : chosen_slope,
                            'GC_INTERCEPT' : chosen_intercept + h}


def coordinate_descent(_obj, _df):
    initial = smf.quantreg('RAW_RDR ~ GC', data=_df).fit(q=.5)
    initial_intercepts = (initial.params['Intercept'], ) + tuple(initial.params['Intercept'] + np.random.uniform(-0.2, 0.2, size=20))
    initial_slope = initial.params['GC']
    return min((coordinate_descent_rep(initial_slope, initial_intercept, _obj) for initial_intercept in initial_intercepts), key=(lambda pair : pair[2]))


def coordinate_descent_rep(initial_slope, initial_intercept, _obj):
    minimize = scipy.optimize.minimize_scalar
    all_objectives = [0, 1e8]
    extract_minimize = (lambda output_minimize : output_minimize.x if all_objectives.append(output_minimize.fun) is None else None)
    solutions = [(None, initial_intercept)]
    get_slope = (lambda _intercept : extract_minimize(minimize((lambda _slope : _obj(_slope, _intercept)), method='bounded', options={'xatol' : 1e-3}, bounds=(initial_slope - 3, initial_slope + 3))))
    get_intercept = (lambda _slope : (_slope, extract_minimize(minimize((lambda _intercept : _obj(_slope, _intercept)), method='bounded', options={'xatol' : 1e-3}, bounds=(initial_intercept - 3, initial_intercept + 3)))))
    _ = [solutions.append(get_intercept(get_slope(solutions[-1][1]))) for _ in range(5) if not np.isclose(all_objectives[-1], all_objectives[-2])]
    return solutions[-1][0], solutions[-1][1], all_objectives[-1]
