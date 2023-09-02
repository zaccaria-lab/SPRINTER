from utils import *

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture



def infer_clones(cns_g1g2, annotations, clone_threshold=None, lower_bound=0.01, min_no_cells=20, norm_thres=0.98, norm_error=0.02, ploidy_tol=1.2, toplot=True):
    assert set(cns_g1g2['CELL'].unique()) == set(annotations[annotations['IS-S-PHASE']==False]['CELL'].unique())
    table = pd.pivot_table(data=cns_g1g2, index='CELL', columns=['CHR', 'START', 'END'], values='CN_TOT')
    threshold = clone_threshold if clone_threshold is not None else estimate_threshold(table, cns_g1g2, norm_error=norm_error, toplot=toplot)
    clus = infer_threshold_clones(table, threshold)
    
    if toplot:
        table = table.reindex(clus['CELL'].unique(), axis=0)
        sns.clustermap(table, row_cluster=False, col_cluster=False, cmap='RdBu_r', center=2, vmin=1, vmax=6, xticklabels=False, yticklabels=False, rasterized=True,
                       row_colors=clus['CLONE'].map(dict(zip(clus['CLONE'].unique(), sns.color_palette('tab20', clus['CLONE'].nunique())))).values)
        plt.savefig('raw_G1G2_clones.png', dpi=300, bbox_inches='tight')

    clus, normal_clones = select_clones(clus, cns_g1g2, norm_thres, lower_bound, min_no_cells)
    clus, normal_clones, ploidy_ref = find_ploidy_errors(clus, cns_g1g2, normal_clones, threshold, ploidy_tol, norm_error)

    if toplot:
        table = table.loc[clus['CELL'].unique()]
        sns.clustermap(table, row_cluster=False, col_cluster=False, cmap='RdBu_r', center=2, vmin=1, vmax=6, xticklabels=False, yticklabels=False, rasterized=True,
                       row_colors=clus['CLONE'].map(dict(zip(clus['CLONE'].unique(), sns.color_palette('tab20', clus['CLONE'].nunique())))).values)
        plt.savefig('final_G1G2_clones.png', dpi=300, bbox_inches='tight')

    clus = clus[['CELL', 'CLONE']].reset_index(drop=True)
    return clus, normal_clones, ploidy_ref


def estimate_threshold(_table, cns_g1g2, norm_error=0.02, max_error=0.4, max_components=5, dist_quantile=.5, toplot=True):
    table = select_cells_distance(cns_g1g2, _table)
    if len(table) > 20:
        distances = np.concatenate([pdist(table[np.random.choice(len(table), size=20, replace=False)], metric='hamming') for _ in range(500)])
    else:
        distances = pdist(table, metric='hamming')
    min_error = norm_error + norm_error * dist_quantile
    get_comps = (lambda params : filter(lambda p : p[0] > min_error, params) if any(p[0] > min_error for p in params) else params)
    get_thres = (lambda gmm : min(get_comps(list(zip(gmm.means_[:,0], np.sqrt(gmm.covariances_.T[0][0])))), key=(lambda p : p[0])))
    gmm_thres = (lambda X : get_thres(min((GaussianMixture(n_components=n, n_init=20).fit(X) for n in range(1, max_components)), key=(lambda gmm : gmm.bic(X)))))
    get_inter = (lambda params : params[0])
    est_thres = get_inter(gmm_thres(distances.reshape(-1, 1)))

    if toplot:
        plt.figure()
        sns.histplot(distances, bins=100, stat='density')
        plt.axvline(x=est_thres, color='red')
        plt.axvline(x=max(norm_error, min(est_thres, max_error)), color='black')
        plt.title(est_thres)
        plt.savefig('clone_est_threshold.png', dpi=600, bbox_inches='tight')
    
    return max(min_error, min(est_thres, max_error))


def select_cells_distance(cns_g1g2, table, flat_error=.1, reps=50):
    flat_cells = cns_g1g2.sort_values(['CELL', 'CHR', 'START', 'END']).reset_index(drop=True)
    flat_cells['SEG_CHR'] = flat_cells.groupby('CELL')['CHR'].transform(lambda x : x != x.shift(1))
    flat_cells['SEG_CNT'] = flat_cells.groupby('CELL')['CN_TOT'].transform(lambda x : x != x.shift(1))
    flat_cells['SEG'] = (flat_cells['SEG_CHR'] | flat_cells['SEG_CNT']).astype(int).cumsum()
    flat_cells['SEG_RDR'] = flat_cells.groupby(['CELL', 'SEG'])['RDR_RTCORR'].transform('median')
    flat_cells = flat_cells.pivot_table(index='CELL', columns=['CHR', 'START', 'END'], values='SEG_RDR')
    is_normal = (lambda X : (((X * 4).round().astype(int) != 4).sum() / len(X)) <= flat_error)
    flat_cells = flat_cells.index[flat_cells.apply(lambda r : is_normal((r / np.nanmean(r.values)).fillna(1.)), axis=1)]
    cell_ploidies = cns_g1g2[~cns_g1g2['CELL'].isin(flat_cells)]
    if cell_ploidies['CELL'].nunique() > 20:
        cell_ploidies = cell_ploidies.groupby('CELL')['CN_TOT'].apply(lambda X : np.mean(np.mean(np.split(X.sample(len(X) * reps, replace=True).values, reps), axis=1))).sort_values()
        cell_ploidies = cell_ploidies.apply(lambda x : cell_ploidies[(x <= cell_ploidies) & (cell_ploidies <= x * 1.2)].index.to_numpy())
        selected_group = cell_ploidies.loc[cell_ploidies.apply(lambda x : len(x)).idxmax()]
        return table.loc[selected_group].values
    else:
        return table.values


def infer_threshold_clones(table, threshold, _min_threshold=0.01):
    linkage = hierarchy.linkage(table.values, method='weighted', metric='hamming', optimal_ordering=True)
    clus = pd.DataFrame(zip(table.index, hierarchy.fcluster(linkage, t=threshold, criterion='distance'))).rename(columns={0 : 'CELL', 1 : 'CLONE'})
    clus = clus.reset_index().rename(columns={'index' : 'OPTIMAL_ORDERING'}).sort_values(['CLONE', 'OPTIMAL_ORDERING']).reset_index(drop=True)
    return clus


def select_clones(clus, cns_g1g2, norm_thres, lower_bound, min_no_cells):
    clones = clus[['CELL', 'CLONE']].drop_duplicates().merge(cns_g1g2, on='CELL', how='inner').reset_index(drop=True)
    clones = clones.groupby(['CLONE', 'CHR', 'START', 'END'])['CN_TOT'].apply(lambda v : v.astype(int).value_counts().index[0]).reset_index()
    norm_fraction = clones.groupby('CLONE', group_keys=False)['CN_TOT'].apply(lambda X : (X.round().astype(int) == 2).astype(int).sum() / len(X))
    normal_clones = np.array(norm_fraction[norm_fraction >= norm_thres].index.to_list() if (norm_fraction >= norm_thres).any() else [])
    clone_sizes = clus[['CELL', 'CLONE']].drop_duplicates().groupby('CLONE')['CELL'].nunique()
    threshold = max(np.round(clus['CELL'].nunique() * lower_bound), min_no_cells)
    clus = clus[clus['CLONE'].isin(clone_sizes[(clone_sizes >= threshold) | (clone_sizes.index.isin(normal_clones))].reset_index()['CLONE'].unique())]
    if len(normal_clones) > 0:
        normal_cells = cns_g1g2[cns_g1g2['CELL'].isin(clus[clus['CLONE'].isin(normal_clones)]['CELL'].unique())]
        no_normal_cells = normal_cells.groupby('CELL')['CN_TOT'].apply(lambda X : (X.round().astype(int) == 2).astype(int).sum() / len(X))
        no_normal_cells = no_normal_cells[no_normal_cells < norm_thres].index
        clus = clus[~clus['CELL'].isin(no_normal_cells)].reset_index(drop=True)
    return clus.sort_values(['CLONE', 'OPTIMAL_ORDERING']).reset_index(drop=True), normal_clones


def find_ploidy_errors(clus, cns_g1g2, normal_clones, threshold, sensitivity, norm_error, _min_threshold=0.02):
    clones = clus[['CELL', 'CLONE']].dropna().drop_duplicates()
    clone_sizes = clones.groupby('CLONE')['CELL'].nunique().sort_values(ascending=True)
    clone_cells = clones.groupby('CLONE')['CELL'].unique().reindex(clone_sizes.index, axis=0)
    cells_table, clone_table, clone_ploidies = segment_ploidy_cns(clones, cns_g1g2, clone_cells)
    ploidy_ref, flat_clone = find_ploidy_ref(clone_sizes, clone_ploidies, normal_clones, clone_table, norm_error)
    clus = extract_ploidy_errors(clus, normal_clones, flat_clone, clone_sizes, clone_cells, cells_table, clone_ploidies, clone_table, ploidy_ref, threshold, sensitivity)

    if clus['CLONE'].nunique() > 1:
        clone_cells = clus.groupby('CLONE')['CELL'].unique().reindex(clus['CLONE'].unique(), axis=0)
        cells_cns_table = cns_g1g2.pivot_table(index='CELL', columns=['CHR', 'START', 'END'], values='CN_TOT')
        clone_cns_table = clone_cells.apply(lambda cells : cells_cns_table.loc[cells].apply(lambda v : v.value_counts().index[0], axis=0)).astype(int) 
        clone_cns_table = clone_cns_table.reindex(clone_cns_table.index[hierarchy.leaves_list(hierarchy.linkage(clone_cns_table, method='weighted', metric='hamming', optimal_ordering=True))], axis=0)
        clone_diff_inner = (lambda p, q, maxcn : (np.minimum(p, maxcn) != np.minimum(q, maxcn)).sum() / len(p))
        clone_diff = (lambda x : clone_diff_inner(x[0], x[1], np.quantile(np.concatenate(x), .95, method='closest_observation')))
        min_threshold = min(threshold, _min_threshold)
        
        is_diff = (pd.Series(index=clone_cns_table.index, data=zip(clone_cns_table.values, clone_cns_table.shift(1).values)).apply(clone_diff) > min_threshold)
        new_clones = (is_diff | is_diff.index.to_series().isin(normal_clones) | is_diff.index.to_series().shift(1).isin(normal_clones)).cumsum()

        clus['PRE_CLONE'] = clus['CLONE']
        clus['CLONE'] = clus['CLONE'].map(new_clones)
        normal_clones = pd.Series(normal_clones).map(new_clones).values

    return clus.sort_values(['CLONE', 'OPTIMAL_ORDERING']).reset_index(drop=True), normal_clones, ploidy_ref


def segment_ploidy_cns(clones, cns_g1g2, clone_cells):
    cells_clcns = cns_g1g2.merge(clones, how='inner').sort_values(['CELL', 'CHR', 'START', 'END'])
    cells_clcns['SEG_CHR'] = cells_clcns.groupby('CELL')['CHR'].transform(lambda x : x != x.shift(1))
    cells_clcns['SEG_CNT'] = cells_clcns.groupby('CELL')['CN_TOT'].transform(lambda x : x != x.shift(1))
    cells_clcns['SEG'] = (cells_clcns['SEG_CHR'] | cells_clcns['SEG_CNT']).astype(int).cumsum()
    cells_clcns['SEG_RDR'] = cells_clcns.groupby(['CELL', 'SEG'])['RDR_RTCORR'].transform('median')
    cells_table = cells_clcns.pivot_table(index='CELL', columns=['CHR', 'START', 'END'], values='SEG_RDR')
    cells_table = cells_table.apply(lambda r : (r / np.nanmean(r.values)).fillna(1.), axis=1)
    clone_table = clone_cells.apply(lambda cells : cells_table.loc[cells].median(axis=0))
    clone_ploidies = cells_clcns.groupby('CLONE')['CN_TOT'].mean().sort_values()
    # sns.heatmap(cells_table, cmap='RdBu_r', center=1, vmax=2)
    # sns.heatmap(np.round(clone_table * clone_ploidies[23.0]).astype(int), cmap='RdBu_r', center=2, vmax=6)
    return cells_table, clone_table, clone_ploidies


def find_ploidy_ref(clone_sizes, clone_ploidies, normal_clones, clone_table, norm_error):
    ploidy_ref = ((np.maximum(clone_ploidies, clone_ploidies.shift(1)) / np.minimum(clone_ploidies, clone_ploidies.shift(1))).fillna(np.inf) > 1.2).astype(int).cumsum().rename('GROUP').reset_index()
    ploidy_ref = ploidy_ref[~ploidy_ref['CLONE'].isin(normal_clones)].reset_index(drop=True)
    flat_clone = clone_table.apply(lambda X : ((X * 2).round().astype(int) != 2).sum() / len(X), axis=1)
    flat_clone = np.setdiff1d(flat_clone[flat_clone <= norm_error].index, normal_clones)
    ploidy_ref = ploidy_ref[~ploidy_ref['CLONE'].isin(flat_clone)].reset_index(drop=True)
    if len(ploidy_ref) > 0:
        ploidy_ref['PLOIDY'] = ploidy_ref['CLONE'].map(clone_ploidies)
        ploidy_ref['SIZE'] = ploidy_ref['CLONE'].map(clone_sizes)
        ploidy_ref['WEIGHTED_PLOIDY'] = (ploidy_ref['PLOIDY'] * ploidy_ref['SIZE']) / ploidy_ref.groupby('GROUP')['SIZE'].transform('sum')
        ploidy_ref = ploidy_ref.groupby('GROUP').agg({'SIZE' : 'sum', 'WEIGHTED_PLOIDY' : 'sum'})
        ploidy_ref = ploidy_ref['WEIGHTED_PLOIDY'].loc[ploidy_ref['SIZE'].idxmax()]
        return ploidy_ref, flat_clone
    else:
        return 2., flat_clone


def extract_ploidy_errors(clus, normal_clones, flat_clone, clone_sizes, clone_cells, cells_table, clone_ploidies, clone_table, ploidy_ref, threshold, sensitivity):

    def is_clone_transf(tag, ref):
        tsize = min(100, clone_sizes[tag], clone_sizes[ref])
        sample = (lambda clone : np.random.choice(clone_cells.loc[clone], tsize, replace=False))
        to_ploidy = max(clone_ploidies[tag], clone_ploidies[ref])
        tag_sample, ref_sample1, ref_sample2 = np.split(np.round(cells_table.loc[np.concatenate([sample(tag), sample(ref), sample(ref)])] * to_ploidy).astype(int).values, 3)
        tag_dist = scipy.spatial.distance.cdist(tag_sample, ref_sample1, metric='hamming').flatten()
        ref_dist = scipy.spatial.distance.cdist(ref_sample1, ref_sample2, metric='hamming').flatten()
        tag_kde = gaussian_kde(tag_dist)
        tag_mode = scipy.optimize.minimize_scalar((lambda x : -tag_kde.logpdf(x)[0]), bounds=(0, 1)).x
        ref_kde = gaussian_kde(ref_dist)
        ref_mode = scipy.optimize.minimize_scalar((lambda x : -ref_kde.logpdf(x)[0]), bounds=(0, 1)).x
        return (np.median(tag_dist) <= (np.median(ref_dist) * sensitivity)) or \
               (np.mean(tag_dist) <= (np.mean(ref_dist) * sensitivity)) or \
               (tag_mode <= (ref_mode * sensitivity)) or \
               (scipy.stats.median_test(tag_dist, ref_dist)[1] >= (0.01 / (len(clone_sizes)**2))) or \
               (tag_mode <= (threshold * sensitivity))

    is_ploidydiff = (lambda ploidy1, ploidy2 : max(ploidy1, ploidy2) / min(ploidy1, ploidy2) > 1.2)
    is_ploidy_error = (lambda tag : any(is_clone_transf(tag, ref) for ref in clone_table.loc[tag:].iloc[1:].index if is_ploidydiff(clone_ploidies[tag], clone_ploidies[ref])))
    ploidy_errors = [clone for clone, cploidy in clone_ploidies.items() if clone not in normal_clones and \
                                                                           is_ploidydiff(cploidy, ploidy_ref) and \
                                                                           is_ploidy_error(clone)]
    is_normal_ploidy_error = (lambda tag : any(is_clone_transf(tag, ref) for ref in normal_clones if is_ploidydiff(clone_ploidies[tag], clone_ploidies[ref])))
    normal_ploidy_errors = [clone for clone, cploidy in clone_ploidies.items() if clone not in ploidy_errors and \
                                                                                  clone not in normal_clones and \
                                                                                  is_ploidydiff(cploidy, ploidy_ref) and \
                                                                                  is_ploidydiff(cploidy, 2) and \
                                                                                  is_normal_ploidy_error(clone)]
    
    return clus[~clus['CLONE'].isin(np.concatenate((ploidy_errors, normal_ploidy_errors, flat_clone)))].sort_values(['CLONE', 'OPTIMAL_ORDERING']).reset_index(drop=True)

