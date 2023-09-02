from utils import *

from callcn import infer_cns_hmm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor



def assign_s_clones(data, _rtprofile, annotations, cn_g1g2, clones_g1g2, normal_clones, 
                    rescue_threshold=None, prop_sphase=0.7, max_norm_error=0.02, max_error=0.5, combine_cns=10, frac_covered=0.2, jobs=1):
    clones, clone_prevalence, max_clone_error, normal_error, normal_clones, selected_clones = process_clones(cn_g1g2, clones_g1g2, normal_clones, max_error)
    assert np.all(np.in1d(clones_g1g2['CELL'].unique(), annotations['CELL'].unique()))
    assert np.all(np.in1d(cn_g1g2['CELL'].unique(), _rtprofile['CELL'].unique()))
    assert np.all(np.in1d(annotations['CELL'].unique(), _rtprofile['CELL'].unique())) and np.all(np.in1d(_rtprofile['CELL'].unique(), annotations['CELL'].unique()))
    cells_to_assign = np.setdiff1d(annotations['CELL'].unique(), clones_g1g2['CELL'].unique())
    assign_data = prepare_bin_global(data, _rtprofile, cn_g1g2, clones, cells_to_assign)
    assert np.all(np.in1d(annotations[annotations['IS-S-PHASE']==True]['CELL'], cells_to_assign))
    cn_assignedcells, assignments = (None, None)
    with Manager() as manager:
        shared_cns = manager.list()
        shared_clones = manager.list()
        with Pool(processes=jobs, 
                  initializer=init_assign_cells, 
                  initargs=(manager.dict({cell : celldf for cell, celldf in assign_data.groupby('CELL')}),
                            manager.dict({clone : clodf for clone, clodf in clones.groupby('CLONE')}),
                            manager.dict(clone_prevalence.to_dict()),
                            shared_cns,
                            shared_clones,
                            combine_cns,
                            frac_covered,
                            manager.dict(max_clone_error.to_dict()),
                            normal_clones,
                            max(normal_error, max_norm_error))) \
        as pool:
            bar = ProgressBar(total=len(cells_to_assign), length=30, verbose=False)
            progress = (lambda e : bar.progress(advance=True, msg="Cell {}".format(e)))
            bar.progress(advance=False, msg="Started")
            _ = [cell for cell in pool.imap_unordered(assign_cells, cells_to_assign) if progress(cell)]
        cn_assignedcells = pd.concat(shared_cns)
        assignments = pd.DataFrame([r for r in shared_clones])
    assert set(cn_assignedcells['CELL'].unique()) == set(cells_to_assign)
    assert 'CLONE' not in cn_assignedcells.columns
    cn_all = pd.concat([cn_g1g2[cn_g1g2['CELL'].isin(clones_g1g2['CELL'].unique())], cn_assignedcells], axis=0, ignore_index=True)
    clones_all = rescue_assignments(assignments, normal_clones, annotations, prop_sphase, selected_clones, clones_g1g2, normal_error, max_norm_error)
    clones_all['IS_REASSIGNED'] = clones_all['CELL'].isin(cells_to_assign)
    assert set(_rtprofile['CELL'].unique()) == set(cn_all['CELL'].unique()) and set(_rtprofile['CELL'].unique()) == set(clones_all['CELL'].unique())
    return cn_all, clones_all


def prepare_bin_global(rawdata, rtprofile, cn_g1g2, clones, cells_to_assign):
    df = rtprofile[rtprofile['CELL'].isin(cells_to_assign)][['CELL', 'CHR', 'BIN_REPINF', 'MERGED_CN_STATE', 'RDR_RTCORR']].assign(FRAC_COVERED=1)
    df = df.merge(rawdata[rawdata['CELL'].isin(cells_to_assign)][['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_REPINF', 'BIN_GLOBAL', 'NORM_COUNT', 'COUNT']],
                  on=['CELL', 'CHR', 'BIN_REPINF'], how='outer')
    columns = ['CELL', 'CHR', 'BIN_GLOBAL', 'START', 'END', 'GENOME', 'NORM_COUNT', 'COUNT', 'MERGED_CN_STATE', 'RDR_RTCORR', 'COUNT_RTCORR', 'FRAC_COVERED']
    df['FRAC_COVERED'] = df['FRAC_COVERED'].fillna(0).astype(int)
    median_counts = rtprofile[rtprofile['CELL'].isin(cells_to_assign)].groupby('CELL')['COUNT'].median()
    df['COUNT_RTCORR'] = df['RDR_RTCORR'] * df['CELL'].map(median_counts)
    df = df[columns].groupby(['CELL', 'CHR', 'BIN_GLOBAL']).agg({'START' : 'min',
                                                                 'END' : 'max',
                                                                 'GENOME' : 'min',
                                                                 'NORM_COUNT' : 'sum',
                                                                 'COUNT' : 'sum',
                                                                 'MERGED_CN_STATE' : (lambda v : v.value_counts(dropna=False).index[0]),
                                                                 'RDR_RTCORR' : 'median',
                                                                 'COUNT_RTCORR' : 'sum',
                                                                 'FRAC_COVERED' : 'mean'})\
                                                         .reset_index()\
                                                         .sort_values(['CELL', 'CHR', 'START', 'END'])\
                                                         .reset_index(drop=True)
    assert not clones['CN_CLONE'].isna().any()
    assert df[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)\
           .equals(cn_g1g2[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
    assert df[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True)\
           .equals(clones[['CHR', 'START', 'END']].drop_duplicates().sort_values(['CHR', 'START', 'END']).reset_index(drop=True))
    return df


def process_clones(cn_g1g2, clones_g1g2, normal_clones, max_error):
    selected_clones = clones_g1g2[['CELL', 'CLONE']].drop_duplicates()
    clone_prevalence = selected_clones.groupby('CLONE')['CELL'].nunique() / len(selected_clones)
    selected_clones_cn = selected_clones.merge(cn_g1g2, on='CELL', how='left')
    selected_clones_cn['CN_CLONE'] = selected_clones_cn.groupby(['CLONE', 'CHR', 'START', 'END'])['CN_TOT'].transform(lambda v : v.value_counts().index[0])
    selected_clones_cn['CN_CLMED'] = selected_clones_cn.groupby(['CLONE', 'CHR', 'START', 'END'])['CN_TOT'].transform('mean')
    selected_clones_cn['CL_CN_ER'] = (selected_clones_cn['CN_CLONE'] != selected_clones_cn['CN_TOT']).astype(int)
    selected_clones['CLONE_ERROR'] = selected_clones['CELL'].map(selected_clones_cn.groupby('CELL')['CL_CN_ER'].apply(lambda err : err.sum() / len(err)))
    max_clone_error = selected_clones.groupby('CLONE')['CLONE_ERROR'].apply(lambda clone_error : np.quantile(clone_error, 0.99, method='closest_observation')).clip(0., max_error)
    clones = selected_clones_cn[['CLONE', 'CHR', 'START', 'END', 'CN_CLONE', 'CN_CLMED']].drop_duplicates().sort_values(['CLONE', 'CHR', 'START', 'END']).reset_index(drop=True)
    if len(normal_clones) == 0:
        norclone = clones['CLONE'].max() + 1
        normal_clones = np.array([norclone])
        assert norclone not in clones['CLONE'].unique() and norclone not in clones_g1g2['CLONE'].unique()
        addnor = clones[['CHR', 'START', 'END']].drop_duplicates().reset_index(drop=True)
        addnor['CLONE'] = norclone
        addnor['CN_CLONE'] = 2
        addnor['CN_CLMED'] = 2
        clones = pd.concat((clones, addnor), axis=0, ignore_index=True).sort_values(['CLONE', 'CHR', 'START', 'END']).reset_index(drop=True)
    normal_error = selected_clones[selected_clones['CLONE'].isin(normal_clones)]['CLONE_ERROR'].max() if len(normal_clones) > 0 else 0.
    return clones, clone_prevalence, max_clone_error, normal_error, normal_clones, selected_clones


def init_assign_cells(_data, _clones, _prevalence, _shared, _shared_clones, _combine, _frac_covered, _maxdiff, _normal_clones, _normal_error):
    global DATA, CLONES, PREVALENCE, SHARED, SHAREDCLONES, COMBINE, FRACCOVERED, MAXDIFF, NORMALCLONES, NORMALERROR
    DATA = _data
    CLONES = _clones
    PREVALENCE = _prevalence
    SHARED = _shared
    SHAREDCLONES = _shared_clones
    COMBINE= _combine
    FRACCOVERED = _frac_covered
    MAXDIFF = _maxdiff
    NORMALCLONES = _normal_clones
    NORMALERROR = _normal_error


def assign_cells(cell):
    globdf = DATA[cell].sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
    celldf = globdf[~pd.isnull(globdf['RDR_RTCORR'])].reset_index(drop=True)
    celldf = celldf[celldf['FRAC_COVERED'] >= FRACCOVERED].reset_index(drop=True)
    assert not celldf['RDR_RTCORR'].isna().any()
    assert not celldf['COUNT'].isna().any()
    norm = (lambda v : v / v.mean())
    celldf['RDR_RTCORR'] = norm(celldf['RDR_RTCORR'].clip(0, 3))
    assert not celldf['RDR_RTCORR'].isna().any()
    celldf['COUNT_RTCORR'] = celldf['RDR_RTCORR'] * celldf['COUNT'].median()

    def clone_difference(clone):
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            resdf = celldf.copy()
            clodf = CLONES[clone].reset_index(drop=True)
            resdf['CN_TOT'] = infer_cns_fixed(resdf, ploidy=np.mean([clodf.sample(len(clodf), replace=True)['CN_CLONE'].mean() for _ in range(50)]), ispoisson=False)

            resdf = resdf.merge(clodf.assign(CELL=cell), on=['CELL', 'CHR', 'START', 'END'], how='inner').sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
            loglk = est_clone_loglk(resdf)
            cell_diff = (resdf['CN_TOT'].astype(int) != resdf['CN_CLONE'].astype(int)).astype(int).sum() / len(resdf)

            resdf = resdf[['CELL', 'CHR', 'START', 'END', 'CN_TOT']].merge(globdf, on=['CELL', 'CHR', 'START', 'END'], how='outer').sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
            resdf = resdf.merge(clodf.assign(CELL=cell), on=['CELL', 'CHR', 'START', 'END'], how='outer').sort_values(['CHR', 'START', 'END']).reset_index(drop=True)
            resdf['CN_TOT'] = resdf['CN_TOT'].where(~pd.isnull(resdf['CN_TOT']), resdf['CN_CLONE'])
            assert resdf[['CHR', 'START', 'END']].sort_values(['CHR', 'START', 'END']).reset_index(drop=True).equals(
                   clodf[['CHR', 'START', 'END']].sort_values(['CHR', 'START', 'END']).reset_index(drop=True))

            assert (~pd.isnull(resdf['CN_TOT']).any()) and (~pd.isnull(resdf['CN_CLONE']).any())
            return resdf[['CELL', 'CHR', 'START', 'END', 'GENOME', 'BIN_GLOBAL', 'COUNT', 'MERGED_CN_STATE', 'RDR_RTCORR', 'CN_TOT']], \
                   cell_diff, \
                   loglk + (np.log(PREVALENCE[clone]) if clone in PREVALENCE else min(PREVALENCE.values()))

    marginals = [clone_difference(clone) + (clone, MAXDIFF[clone] if clone not in NORMALCLONES else NORMALERROR) for clone in CLONES.keys()]
    candidates = list(filter(lambda x : x[1] <= x[-1], marginals))
    chosen = max(candidates, key=(lambda x : x[2])) if len(candidates) > 0 else max(marginals, key=(lambda x : x[2]))
    assert chosen[3] in CLONES.keys()
    assert chosen[1] <= chosen[4] or len(candidates) == 0
    SHARED.append(chosen[0])
    SHAREDCLONES.append({'CELL' : cell, 'CLONE' : chosen[3], 'MAX_CLONE_DIST' : chosen[4], 'DIST' : chosen[1], 'POST' : chosen[2], 'IS_MAPPED' : len(candidates) > 0})
    return cell


def infer_cns_fixed(data, ploidy, ispoisson=False):
    assert not pd.isnull(data['RDR_RTCORR']).any()
    assert not pd.isnull(data['COUNT']).any()
    var = data.groupby('MERGED_CN_STATE')['RDR_RTCORR'].apply(lambda x : x.clip(*scipy.stats.norm.interval(0.99, *scipy.stats.norm.fit(x))).var(ddof=1)).mean()
    trans = data['MERGED_CN_STATE'].nunique() / len(data)
    norm = (lambda v : v / v.mean())
    if not ispoisson:
        rdrs = norm(data['RDR_RTCORR'].to_numpy())
    else:
        scale = data['COUNT'].median()
        rdrs = norm((data['RDR_RTCORR'] * scale).round().astype(int).to_numpy())
    return infer_cns_hmm(rdrs, gamma=ploidy, var=var, trans=trans, ispoisson=ispoisson)[2]


def est_clone_loglk(_data, ispoisson=False):
    assert not pd.isnull(_data['RDR_RTCORR']).any()
    assert not pd.isnull(_data['CN_CLONE']).any()
    assert not pd.isnull(_data['COUNT']).any()
    data = _data[['RDR_RTCORR', 'CN_CLONE'] if not ispoisson else ['RDR_RTCORR', 'CN_CLONE', 'COUNT']].reset_index(drop=True)
    norm = (lambda v : v / v.mean())
    if not ispoisson:
        rdrs = norm(data['RDR_RTCORR'].to_numpy())
    else:
        scale = data['COUNT'].median()
        rdrs = (data['RDR_RTCORR'] * scale).round().astype(int).to_numpy()
    ploidy = data['CN_CLONE'].mean()
    llkdf = pd.DataFrame({'RDR' : rdrs, 'CN' : data['CN_CLONE'].to_numpy()})
    get_llk = (lambda X, mean : scipy.stats.norm.logpdf(X, loc=mean, scale=scipy.stats.norm.fit(X, floc=mean)[1]).sum())
    return llkdf.groupby('CN')['RDR'].apply(lambda x : get_llk(x, x.name / ploidy)).sum()


def rescue_assignments(assignments, normal_clones, annotations, prop_sphase, selected_clones, clones_g1g2, normal_error, max_norm_error):
    normal_cells = np.union1d(clones_g1g2[clones_g1g2['CLONE'].isin(normal_clones)]['CELL'].unique(),
                              assignments[assignments['CLONE'].isin(normal_clones)]['CELL'].unique())
    s_cancer_cells = annotations[(annotations['IS-S-PHASE']==True) & (~annotations['CELL'].isin(normal_cells))]['CELL'].unique()
    g_cancer_cells = annotations[(annotations['IS-S-PHASE']==False) & (~annotations['CELL'].isin(normal_cells))]['CELL'].unique()
    assert set(np.union1d(np.union1d(s_cancer_cells, g_cancer_cells), normal_cells)) == set(annotations['CELL'].unique())
    dist_g_cells = selected_clones[selected_clones['CELL'].isin(g_cancer_cells)][['CELL', 'CLONE', 'CLONE_ERROR']].rename(columns={'CLONE_ERROR' : 'DIST'})
    dist_g_cells = pd.concat((dist_g_cells, assignments[assignments['CELL'].isin(g_cancer_cells)][['CELL', 'CLONE', 'DIST']]), axis=0, ignore_index=True)
    dist_g_cells = dist_g_cells.sort_values('DIST', ascending=True).reset_index(drop=True)
    assert set(dist_g_cells['CELL'].unique()) == set(g_cancer_cells)
    dist_s_cells = assignments[assignments['CELL'].isin(s_cancer_cells)][['CELL', 'CLONE', 'DIST']].sort_values('DIST', ascending=True).reset_index(drop=True)
    assert set(dist_s_cells['CELL'].unique()) == set(s_cancer_cells)
    assign_cancer_cells = pd.concat((dist_g_cells[['CELL', 'CLONE']].head(int(round(len(dist_g_cells) * prop_sphase))),
                                     dist_s_cells[['CELL', 'CLONE']].head(int(round(len(dist_s_cells) * prop_sphase)))), axis=0, ignore_index=True)
    assign_cancer_cells = pd.concat((assign_cancer_cells,
                                     pd.DataFrame({'CELL' : np.setdiff1d(np.union1d(g_cancer_cells, s_cancer_cells), assign_cancer_cells['CELL'].unique()), 'CLONE' : np.nan})), 
                                     axis=0, ignore_index=True)
    assignments_normals = assignments[assignments['CLONE'].isin(normal_clones)].reset_index(drop=True)
    assign_normal_cells = pd.concat((clones_g1g2[clones_g1g2['CELL'].isin(normal_cells)][['CELL', 'CLONE']],
                                     assignments_normals[assignments_normals['DIST'] <= max(normal_error, max_norm_error)][['CELL', 'CLONE']]), axis=0, ignore_index=True)
    assign_normal_cells = pd.concat((assign_normal_cells,
                                     pd.DataFrame({'CELL' : np.setdiff1d(normal_cells, assign_normal_cells['CELL'].unique()), 'CLONE' : np.nan})), 
                                     axis=0, ignore_index=True)
    all_assignments = pd.concat((assign_cancer_cells, assign_normal_cells), axis=0, ignore_index=True)
    assert set(all_assignments['CELL'].unique()) == set(annotations['CELL'].unique())
    return all_assignments


def infer_rescue_threshold(distances, dist_quantile=.05, max_components=5):
    get_thres = (lambda gmm : min(list(zip(gmm.means_[:,0], np.sqrt(gmm.covariances_.T[0][0]))), key=(lambda p : p[0])))
    gmm_thres = (lambda X : get_thres(min((GaussianMixture(n_components=n, n_init=20).fit(X) for n in range(1, max_components)), key=(lambda gmm : gmm.bic(X)))))
    get_inter = (lambda params : scipy.stats.norm.interval(dist_quantile, loc=params[0], scale=params[1])[1])
    est_thres = get_inter(gmm_thres(distances.reshape(-1, 1)))
    plt.figure()
    sns.histplot(distances, bins=100, stat='density')
    plt.axvline(x=est_thres, color='red')
    plt.title(est_thres)
    plt.savefig('threshold_distance_rescue.png', dpi=300, bbox_inches='tight')
    return est_thres
