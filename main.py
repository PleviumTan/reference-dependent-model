from utils import *
from render import *
from disc import *

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy
import os
from tqdm import tqdm

def tukey_hsd( lst, ind, n ):
    data_arr = np.hstack( lst )
    ind_arr = np.repeat(ind, n)
    #print(pairwise_tukeyhsd(data_arr,ind_arr))


def compare_full(parsed_log):
    ll = len(parsed_log)
    y = [q['satisfaction'] for q in parsed_log]

    for mn in METRIC_LIST:
        if mn in METRIC_PARAMETER_BOUND:
            p_1 = METRIC_PARAMETER_BOUND[mn][0]
            p_2 = METRIC_PARAMETER_BOUND[mn][1]
            step = METRIC_PARAMETER_BOUND[mn][2]
        else:
            p_1 = -1
            p_2 = -1
            step = -1
        p_st = tune_metric_parameter(parsed_log, mn, p_1, p_2, step)
        #p_st = DEFAULT_PARAMETER[mn]

        metric = get_my_metric(mn, p_st)
        x_map = get_x(parsed_log, metric)
        corr_map = {}
        for a_mode in x_map:
            #corr = scipy.stats.spearmanr(x_map[a_mode], y)
            corr, p = scipy.stats.kendalltau(x_map[a_mode], y)
            lc, rc = get_CI(corr, ll, 0.05)

            #corr = scipy.stats.pearsonr(x_map[a_mode], y)[0]
            corr_map[a_mode] = [corr, lc, rc, p]

        print(12 * '>')
        print(metric.name)
        for a_mode in corr_map:
            res = corr_map[a_mode]
            print('Mode - %s Corr: %.3f 95%% CI: (%.3f, %.3f)  p:%.4f' % (a_mode, res[0], res[1], res[2], res[3]))




def best_aggregation(parsed_log):
    ll = len(parsed_log)
    for mn in RM_LIST:
        print("Metric: %s" % (mn))
        metric = get_my_metric(mn, -1)
        y = [q['satisfaction'] for q in parsed_log]
        x_map = get_x(parsed_log, metric)
        corr_map = {}
        for a_mode in x_map:
            #corr = scipy.stats.spearmanr(x_map[a_mode], y)
            corr, p = scipy.stats.kendalltau(x_map[a_mode], y)
            lc, rc = get_CI(corr, ll, 0.05)
            #corr = scipy.stats.pearsonr(x_map[a_mode], y)[0]
            print("Aggregation: %s Corr: %.3f p:%.10f" % (a_mode, corr, p))


def metric_compare(parsed_log):
    metric_corr_all = {}
    for i in tqdm(range(0, 50)):
        #print('Iteration: %d' % (i))
        train_set, test_set = train_test_split(parsed_log, test_size=0.4, random_state=(i * 514 + 810))

        for mn in METRIC_LIST:
            if mn not in metric_corr_all:
                metric_corr_all[mn] = {}
            if mn in METRIC_PARAMETER_BOUND:
                p_1 = METRIC_PARAMETER_BOUND[mn][0]
                p_2 = METRIC_PARAMETER_BOUND[mn][1]
                step = METRIC_PARAMETER_BOUND[mn][2]
            else:
                p_1 = -1
                p_2 = -1
                step = -1
            p_st = tune_metric_parameter(train_set, mn, p_1, p_2, step)
            #p_st = DEFAULT_PARAMETER[mn]

            metric = get_my_metric(mn, p_st)
            y = [q['satisfaction'] for q in test_set]
            x_map = get_x(test_set, metric)
            corr_map = {}
            for a_mode in x_map:
                #corr = scipy.stats.spearmanr(x_map[a_mode], y)
                corr, p = scipy.stats.kendalltau(x_map[a_mode], y)
                #lc, rc = get_CI(corr, ll, 0.05)
                #corr = scipy.stats.pearsonr(x_map[a_mode], y)[0]
                corr_map[a_mode] = (corr, p)
                if a_mode not in metric_corr_all[mn]:
                    #print("%s %s" % (mn, a_mode))
                    metric_corr_all[mn][a_mode] = []
                metric_corr_all[mn][a_mode].append(corr)

            #print(12 * '>')
            #print(metric.name)
            '''
            for a_mode in corr_map:
                res = corr_map[a_mode]
                print('Mode - %s Corr: %.3f' % (a_mode, res[0]))
            '''

    for m in metric_corr_all:
        print("Metric: %s" % (m))
        for a in metric_corr_all[m]:
            res = metric_corr_all[m][a]
            #print(res)
            print("Aggregation: %s Mean: %.3f Std:%.3f" % (a, np.mean(res), np.std(res)))

    for rm in RM_LIST:
        lst = [metric_corr_all[rm]['ERG']]
        ind = [rm]
        n = len(metric_corr_all[rm]['ERG'])
        print('*'*6 + rm + '*'*6)
        for baseline in BASELINE_LIST:
            #lst.append(metric_corr_all[baseline][DEFAULT_A_MODE[baseline]])
            #ind.append(baseline)
            stat, p = scipy.stats.ttest_rel(metric_corr_all[rm]['ERG'],
                                            metric_corr_all[baseline][DEFAULT_A_MODE[baseline]])
            print("%s - ERG vs %s t: %.3f p-value: %.4f" % (rm, baseline, stat, p))

        #tukey_hsd( (np.array(l) for l in lst), ind , n)

    '''
    for rm in RM_LIST:
        print('*'*6 + rm + '*'*6)
        for baseline in BASELINE_LIST:
            stat, p = scipy.stats.ttest_rel(metric_corr_all[rm]['ETG'],
                                            metric_corr_all[baseline][DEFAULT_A_MODE[baseline]])
            print("%s - ETG vs %s t: %.3f p-value: %.4f" % (rm, baseline, stat, p))
    '''

if __name__ == '__main__':


    thuir1 = thuir1_parse(THUIR1_DIR, THUIR1_SCORE_FILE, THUIR1_REL_FILE)
    fsd = tiangong_fsd_parse(FSD_DIR)
    kdd19 = parse_kdd(KDD_LOG_FILE, KDD_USEFUL_FILE)
    qref = tiangong_qref_parse(QREF_DIR)
    # RQ1
    best_aggregation(fsd)

    # RQ2

    print('#'*16 + '\n' + 'THUIR1\n'+'#'*16)
    metric_compare(thuir1)
    #print('#'*16 + '\n' + 'TianGong - FSD\n'+'#'*16)
    #metric_compare(fsd)
    print('#'*16 + '\n' + 'KDD19\n'+'#'*16)
    metric_compare(kdd19)
    print('#'*16 + '\n' + 'TianGong - Qref\n'+'#'*16)
    metric_compare(qref)



    #RQ3

    tiangong_lcc = list(filter(low_cost, qref))
    print('Low Cognitive Load: '+ str(len(tiangong_lcc)) + ' records.')
    compare_full(tiangong_lcc)

    tiangong_mcc = list(filter(mid_cost, qref))
    print('Mid Cognitive Load: '+ str(len(tiangong_mcc)) + ' records.')
    compare_full(tiangong_mcc)

    tiangong_hcc = list(filter(high_cost, qref))
    print('High Cognitive Load: '+ str(len(tiangong_hcc)) + ' records.')
    compare_full(tiangong_hcc)


    tiangong_le = list(filter(low_expertise, qref))
    print('Low Expertise '+ str(len(tiangong_le)) + ' records.')
    compare_full(tiangong_le)
    tiangong_he = list(filter(high_expertise, qref))
    print('High Expertise '+ str(len(tiangong_he)) + ' records.')
    compare_full(tiangong_he)

    # Draw figures
    '''
    # figures
    render_curve()

    render_corr(qref,'tiangong')
    render_corr(thuir1,'thuir1')
    render_corr(kdd19,'kdd19')
    '''
    #matrix = get_agreement(qref)
    #render_agreement(matrix, 'tiangong')
    #matrix = get_agreement(thuir1)
    #render_agreement(matrix, 'thuir1')
    #matrix = get_agreement(kdd19)
    #render_agreement(matrix, 'kdd19')
