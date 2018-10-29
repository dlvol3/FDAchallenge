#%%
# Generating the allinone table
# Flat the random forest classifiers
# Yue Zhang <yue.zhang@lih.lu>
# Oct 2018

import numpy as np
import pandas as pd
import time


def timing(func):
    def wrap(*args, **kw):
        print('<function name: %s>' % func.__name__)
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[time elapsed: %d s]' % (time2-time1))
        return ret
    return wrap

# TODO add the regression support
# TODO add the weights of the depth of the nodes
# TODO add the defined default the value
@timing
def flatforest(rf, testdf):
    try:
        tree_infotable = pd.DataFrame()
        raw_hits = pd.DataFrame()
        predictlist_for_all = pd.DataFrame()

        for t in range(rf.n_estimators):
            # Generate the info table for trees

            # Preparation

            # Node index # Count from leftleft first /list
            nodeIndex = list(range(0, rf.estimators_[t].tree_.node_count, 1))
            # Node index forest level
            nodeInForest = list(map(lambda x: x + rf.decision_path(testdf)[1].item(t), nodeIndex))
            # lc # left children of each node, by index ^ /ndarray 1D
            lc = rf.estimators_[t].tree_.children_left
            # rc # right children of each node, by index ^ /ndarray 1D
            rc = rf.estimators_[t].tree_.children_right
            # Proportion of sample in each nodes  /ndarray +2D add later
            # TODO if the pv info is needed for the weighted GS score, re-calculate. No need to add it into the table.
            pv = rf.estimators_[t].tree_.value
            # Feature index, by index /1d array
            featureIndex = rf.estimators_[t].tree_.feature
            # Feature threshold, <= %d %threshold
            featureThreshold = rf.estimators_[t].tree_.threshold
            # Gini impurity of the node, by index
            gini = rf.estimators_[t].tree_.impurity
            # Tree index
            treeIndex = t+1
            testlist = pd.DataFrame(
                {'node_index': nodeIndex,
                 'left_c': lc,
                 'right_c': rc,
                 'feature_index': featureIndex,
                 'feature_threshold': featureThreshold,
                 'gini': gini,
                 'tree_index': treeIndex,
                 'nodeInForest': nodeInForest
                 })

            # Calculation of the default gini gain
            gslist = list()
            nodetype = list()
            for ii in range(rf.estimators_[t].tree_.node_count):
                if testlist.loc[:, 'feature_index'][ii] == -2:
                    gslist.append(-1)
                    nodetype.append("leaf_node")
                    continue  # Next if node is leaf

                ri = testlist.loc[:, 'right_c'][ii]  # right child index of node i
                li = testlist.loc[:, 'left_c'][ii]  # left child index of node i

                gs_index = testlist.loc[:, 'gini'][ii] \
                    - np.sum(pv[li])/np.sum(pv[ii])*testlist.loc[:, 'gini'][li] \
                    - np.sum(pv[ri])/np.sum(pv[ii])*testlist.loc[:, 'gini'][ri]

                gslist.append(gs_index)
                nodetype.append("decision_node")

            testlist['GS'] = pd.Series(gslist).values
            testlist['node_type'] = pd.Series(nodetype).values

            tree_infotable = pd.concat([tree_infotable, testlist])
        print("Forest %s flatted, matrix generate with %d rows and %d columns" % (rf, tree_infotable.shape[0],
                                                                                  tree_infotable.shape[1]))
        for s_index in range(rf.decision_path(testdf)[0].indptr.shape[0] - 1):  # Loop on samples for prediction
            sample_ceiling = rf.decision_path(testdf)[0].indptr[s_index + 1]  # The ceiling hit index of the current s
            sample_floor = rf.decision_path(testdf)[0].indptr[s_index]
            hitall = pd.DataFrame()
            predictlist = list()   # Store the predictions among the forest for a certain sample
            treelist = list()
            samplelist = s_index
            for ttt in range(rf.n_estimators):
                pred_s_t = rf.estimators_[ttt].predict(testdf)[s_index]
                predictlist.append(pred_s_t)
                treelist.append(ttt)
            predictlist_for_sample = pd.DataFrame(
                {'prediction': predictlist,
                 'tree index': treelist,
                 'sample': samplelist
                 })
            predictlist_for_sample['matching'] = np.where(predictlist_for_sample['prediction'] ==
                                                          rf.predict(testdf)[predictlist_for_sample['sample']],
                                                          'match', 'not_matching')
            predictlist_for_all = pd.concat([predictlist_for_all, predictlist_for_sample])

            for hit_index in range(sample_floor, sample_ceiling):  # Loop through the hits of the current sample
                hit = tree_infotable.loc[tree_infotable['nodeInForest'] == rf.decision_path(testdf)[0].indices[hit_index],
                            ['feature_index', 'GS', 'tree_index','feature_threshold']]
                hit['sample_index'] = pd.Series(s_index).values
                hitall = pd.concat([hitall, hit])
            raw_hits = pd.concat([raw_hits, hitall])

        df = list()
        df.extend((tree_infotable, raw_hits, predictlist_for_all))
        print("All node used for predicting samples extracted")
        return df
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)



#%%
# Generating the allinone table
# Take the output of flatforest() to generate the network ready table for the NERF progress
# Yue Zhang <yue.zhang@lih.lu>
# Oct 2018


import pandas as pd
import time


def timing(func):
    def wrap(*args, **kw):
        print('<function name: {0}>'.format(func.__name__))
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[timecosts: {0} s]'.format(time2-time1))
        return ret
    return wrap


@timing
# nerftab function for generating pairs of the 'correct' decision features
def nerftab(df_ff):
    # All possible pairs generator
    # TODO give the tree a index?
    # TODO Same to previous one, give sample a index to loop with

    try:
        list_of_single_decisions = pd.DataFrame()
        list_of_all_decision_pairs = pd.DataFrame()
        for psample in range(max(df_ff[1].loc[:, 'sample_index']) + 1):  # Loop on predict samples
            t_in_psample = df_ff[2].loc[(df_ff[2]['matching'] == 'match') & (df_ff[2]['sample'] == psample),'tree index']
            # Trees with the 'correct' prediction during the prediction process of the sample psample

            t_p_psample = df_ff[1].loc[(df_ff[1]['tree_index'].isin(t_in_psample + 1)) &
                                     (df_ff[1]['sample_index'] == psample) & (df_ff[1]['feature_index'] != -2), ]
            # t_in_sample + 1 to make sure the alignment between df_ff[] tree index
            # For each sample, get the 'correct' trees with decision paths, remove the leaf nodes

            single_decisions_sample = pd.DataFrame()
            all_decision_pairs_sample = pd.DataFrame()
            for itree in df_ff[2].loc[(df_ff[2]['matching'] == 'match') & (df_ff[2]['sample'] == psample), 'tree index']:
                nodes_cand = t_p_psample.loc[t_p_psample['tree_index'] == (itree + 1), ]  # Of each tree!
                if nodes_cand.shape[0] == 1:
                    #  If-statement for the situation where only one decision node presents
                    single_decisions_sample = pd.concat([single_decisions_sample, nodes_cand])
                    continue
                else:
                    pairsofonetree = pd.DataFrame()
                    for i in range(nodes_cand.shape[0]):
                        pairs_inside = pd.DataFrame()
                        for j in range(i + 1, nodes_cand.shape[0]):
                            if nodes_cand.iloc[i, 0] <= nodes_cand.iloc[j, 0]:
                                f_i = nodes_cand.iloc[i, 0]  # feature index of feature i, the small one
                                f_j = nodes_cand.iloc[j, 0]
                            elif nodes_cand.iloc[i, 0] > nodes_cand.iloc[j, 0]:
                                f_j = nodes_cand.iloc[i, 0]  # feature index of feature i, assign the small one to i
                                f_i = nodes_cand.iloc[j, 0]
                            gs_i = nodes_cand.iloc[i, 1]  # gs of feature i
                            gs_j = nodes_cand.iloc[j, 1]
                            ft_i = nodes_cand.iloc[i, 3]  # feature threshold of i
                            ft_j = nodes_cand.iloc[j, 3]
                            tr_index = nodes_cand.iloc[i, 2]
                            sp_index = nodes_cand.iloc[i, 4]
                            listofunit = [[f_i, f_j, gs_i, gs_j, ft_i, ft_j, tr_index, sp_index]]
                            dfunit = pd.DataFrame(listofunit,
                                                  columns=['feature_i', 'feature_j', 'GS_i', 'GS_j', 'threshold_i',
                                                           'threshold_j',
                                                           'tree_index', 'sample_index'])
                            pairs_inside = pd.concat([pairs_inside, dfunit])
                        pairsofonetree = pd.concat([pairsofonetree, pairs_inside])
                all_decision_pairs_sample = pd.concat([all_decision_pairs_sample, pairsofonetree])
                # One sample, all the trees^^^
            list_of_single_decisions = pd.concat([list_of_single_decisions, single_decisions_sample])
            list_of_all_decision_pairs = pd.concat([list_of_all_decision_pairs, all_decision_pairs_sample])
        df = list()
        df.extend((list_of_single_decisions, list_of_all_decision_pairs))
        print("NERFtab finished")
        return df
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)



#%%
# calculating the I(Fi, Fj) while generating the edge intensity
# Take the output of nerftab() to generate the network ready table for the NERF progress
# Yue Zhang <yue.zhang@lih.lu>
# Oct 2018

import numpy as np
import pandas as pd
import time
import math
import networkx as nx
import os

def timing(func):
    def wrap(*args, **kw):
        print('<function name: {0}>'.format(func.__name__))
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[timecosts: {0} s]'.format(time2-time1))
        return ret
    return wrap


@timing
def localnerf(nf_ff, local_index):
    try:
        allpairs_local = nf_ff[1].loc[(nf_ff[1]['sample_index'] == local_index, )]
        allpairs_local['GSP'] = (allpairs_local.loc[:, 'GS_i'] + allpairs_local.loc[:, 'GS_j'])
        localtable = allpairs_local.groupby(['feature_i', 'feature_j'], as_index=False)['GSP'].agg([np.size, np.sum]).reset_index()
        localtable['EI'] = localtable.values[:, 3] * localtable.values[:, 2]
        output_local = localtable.loc[:, ['feature_i', 'feature_j', 'EI']]

        return output_local
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)

# This is like 1000+ times faster...
# Try to be smart otherwise you are screwed up


# Sort by value, Dict
def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]


# Define a func for later
@timing
def twonets(outdf, filename, index_of_features='omit', featureref='omit', index1=2, index2=10):
    """
    Purpose
    ---------
    Process the result of the localnerf(), return two networks, one with everything, one with less info
    ---------
    :param outdf: The localnerf() result
    :param filename: the desire filename, or path with name, no suffix
    :param index1: Index for selecting top degree of centrality, default = 2
    :param index_of_features: The original order of all the features, set to false as D
    :param featureref: The target feature reference, e.g., gene symbols, set to false as D
    :param index2: Index for selecting top edge intensity, default = 10
    :return: A list contains five elements, whole network with gene names, degree of all features,
     degreetop selected, eitop delected, sub network with gene names
    """
    if not os.path.exists("output"):
        os.makedirs("output")
    if index_of_features or featureref != 'omit':
        outdf = outdf.replace(index_of_features, featureref)

    else:
        pass
    # export the 'everything' network
    outdf.to_csv(os.getcwd() + '/output/' + filename + "_everything.txt", sep='\t')

    gout = nx.from_pandas_edgelist(outdf, "feature_i", "feature_j", "EI")

    degreecout = nx.degree_centrality(gout)
    # Test save the centrality
    degreecoutdf = pd.DataFrame.from_dict(degreecout, orient="index")
    degreecoutdf.to_csv(os.getcwd() + '/output/' + filename + "_DC.txt", sep='\t')
    # Large to small sorting
    sortdegree = sort_by_value(degreecout)
    # take the top sub of the DC
    degreetop = sortdegree[: int(index1 * math.sqrt(len(sortdegree)))]
    # Large to small sorting, Edge intensity
    outdfsort = outdf.sort_values('EI', ascending=False)

    eitop = outdfsort[: int(index2 * math.sqrt(outdfsort.shape[0]))]

    outdffinal = eitop[eitop['feature_i'].isin(degreetop) & eitop['feature_j'].isin(degreetop)]
    outdffinal.to_csv(os.getcwd() + '/output/' + filename + '_sub.txt', sep='\t')
    outputfunc = list()
    outputfunc.extend((outdf, degreecout, degreetop, eitop, outdffinal))
    print("Processing finished.")
    return outputfunc


