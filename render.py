"""
This file is programmes for figure render
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import *
from config import *
from metrics import *
import scipy
import numpy as np
import matplotlib
from tqdm import tqdm

class RAExample(Metric):

    def __init__(self, ref):
        self.init()
        self.name = 'RA_Example'
        self.ref = ref

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        ref = self.ref
        depth = len(result_list)
        while rank < depth:
            cur_rel = float(result_list[rank])
            # User will definitely stop at the truncation rank
            c = ((2 + rank - cur_rel)/(3 + rank - cur_rel + ref))

            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


def render0():
    x = [i for i in range(1, 11)]

    rel_0 = [0 for i in range(10)]
    rel_1 = [1 for i in range(10)]

    ref0 = RAExample(0)

    ref0.read_serp(rel_0)
    c1 = [c for c in ref0.C]
    ref0.read_serp(rel_1)
    c2 = [c for c in ref0.C]


    ax = plt.gca()

    ax.set_facecolor('#ECEBF0') # Set background-colour
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框

    plt.plot(x, c1, label="r = 0 for each item")
    plt.plot(x, c2, label="r = 1 for each item")

    plt.grid(c='#FFFFFF')
    plt.xticks(color = "#3c3c3c", fontsize=14)
    plt.yticks(color = "#3c3c3c", fontsize=14)

    plt.xlabel('Rank Position', color = "#3c3c3c", fontsize=14)
    plt.ylabel('Continuation Probability', color = "#3c3c3c", fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.title('C(i) When ref = 0 Constantly')
    plt.ylim(0, 1)
    plt.savefig('output/ref0.png', dpi=500, bbox_inches='tight')

    plt.show()


def render1():
    x = [i for i in range(1, 11)]

    rel_0 = [0 for i in range(10)]
    rel_1 = [1 for i in range(10)]

    ref1 = RAExample(1)

    ref1.read_serp(rel_0)
    c1 = [c for c in ref1.C]
    ref1.read_serp(rel_1)
    c2 = [c for c in ref1.C]

    ax = plt.gca()

    ax.set_facecolor('#ECEBF0') # Set background-colour
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框

    plt.plot(x, c1, label="r = 0 for each item")
    plt.plot(x, c2, label="r = 1 for each item")

    plt.grid(c='#FFFFFF')
    plt.xticks(color = "#3c3c3c", fontsize=14)
    plt.yticks(color = "#3c3c3c", fontsize=14)


    plt.xlabel('Rank Position', color = "#3c3c3c", fontsize=14)
    plt.ylabel('Continuation Probability', color = "#3c3c3c", fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.title('C(i) When ref = 1 Constantly')
    plt.ylim(0,1)
    plt.savefig('output/ref1.png', dpi=500, bbox_inches='tight')

    plt.show()


def render2():
    x = [i for i in range(1, 11)]

    rel_0 = [0 for i in range(10)]
    rel_1 = [1 for i in range(10)]

    ref0 = RAExample(0)

    ref0.read_serp(rel_0)
    c1 = [c for c in ref0.V]
    ref0.read_serp(rel_1)
    c2 = [c for c in ref0.V]

    ax = plt.gca()

    ax.set_facecolor('#ECEBF0') # Set background-colour
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框

    plt.plot(x, c1, label="r = 0 for each item")
    plt.plot(x, c2, label="r = 1 for each item")

    plt.grid(c='#FFFFFF')
    plt.xticks(color = "#3c3c3c", fontsize=14)
    plt.yticks(color = "#3c3c3c", fontsize=14)

    plt.xlabel('Rank Position', color = "#3c3c3c", fontsize=14)
    plt.ylabel('View Probability', color = "#3c3c3c", fontsize=14)
    plt.legend( fontsize=14)
    plt.title('V(i) When ref = 0 Constantly')
    plt.ylim(0,1)
    plt.savefig('output/ref2.png', dpi=500, bbox_inches='tight')

    plt.show()


def render3():
    x = [i for i in range(1, 11)]

    rel_0 = [0 for i in range(10)]
    rel_1 = [1 for i in range(10)]

    ref1 = RAExample(1)

    ref1.read_serp(rel_0)
    c1 = [c for c in ref1.V]
    ref1.read_serp(rel_1)
    c2 = [c for c in ref1.V]

    ax = plt.gca()

    ax.set_facecolor('#ECEBF0') # Set background-colour
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框

    plt.plot(x, c1, label="r = 0 for each item")
    plt.plot(x, c2, label="r = 1 for each item")

    plt.grid(c='#FFFFFF')
    plt.xticks(color = "#3c3c3c", fontsize=14)
    plt.yticks(color = "#3c3c3c", fontsize=14)


    plt.xlabel('Rank Position', color = "#3c3c3c", fontsize=14)
    plt.ylabel('View Probability', color = "#3c3c3c", fontsize=14)
    plt.legend( fontsize=14)
    plt.title('V(i) When ref = 1 Constantly')
    plt.ylim(0,1)
    plt.savefig('output/ref3.png', dpi=500, bbox_inches='tight')

    plt.show()


def render_curve():
    render0()
    render1()
    render2()
    render3()


def render_corr(dataset, dataset_name):
    metric_list = [#Prec(), RR(), DCG(2), RBP(0.5), RBP(0.8),  INST(1), INST(5),
                   RAStatic1(), RADynamic1(), RADynamic2(), RADynamic4(), RADynamic3()]
    score_list = [[] for i in range(len(metric_list))]
    for record in dataset:
        rl = record['SERPs'][0]['top10_usefulness']
        for i in range(len(metric_list)):
            metric_list[i].read_serp(rl)
            score_list[i].append(metric_list[i].get_erg())


    corr_map = [[0 for j in range(len(metric_list))]for i in range(len(metric_list))]
    for i in range(len(metric_list)):
        for j in range(len(metric_list)):
            corr_map[i][j] = round(scipy.stats.kendalltau(score_list[i], score_list[j])[0], 3)

    #print(corr_map)
    x_label = [#"Prec", "ERR", "DCG@10", "RBP@Φ=0.5", "RBP@Φ=0.8", "INST@T=1", "INST@T=5",
               "ReDeM-Init", "ReDeM-Max", "ReDeM-End", "ReDeM-Avg", "ReDeM-PE"]
    y_label = x_label

    corr_map = np.array(corr_map)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_map, cmap=plt.get_cmap("Reds_r").reversed())

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label)))
    ax.set_yticks(np.arange(len(y_label)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_label)):
        for j in range(len(y_label)):
            text = ax.text(j, i, corr_map[i, j],
                ha="center", va="center", color="k", fontsize = 8)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson\'s r", rotation=-90, va="bottom")
    ax.set_title("Pearson's r of Each Measure Pair")

    fig.tight_layout()
    plt.savefig('output/'+dataset_name+'_corr_pair.png', dpi=600, bbox_inches='tight')
    plt.show()


def get_agreement(dataset):
    metric_list = [
        #Prec(), RR(), DCG(2), RBP(0.5), RBP(0.8),  INST(1), INST(5),
                   RAStatic1(), RADynamic1(), RADynamic2(), RADynamic4(), RADynamic3()]
    score_list = [[] for i in range(len(metric_list))]
    aggreement_matrix = [[0 for j in range(len(metric_list))] for i in range(len(metric_list))]

    for record in dataset:
        rl = record['SERPs'][0]['top10_usefulness']
        for i in range(len(metric_list)):
            metric_list[i].read_serp(rl)
            if i != 1:
                score_list[i].append(metric_list[i].get_erg())
            else:
                score_list[i].append(metric_list[i].get_err())
    cnt = 0
    pair_diff = [[]for i in range(len(metric_list))]
    print('Generating pair difference')
    for i in range(len(dataset)):
        for j in range(i):
            cnt += 1
            for k in range(len(metric_list)):
                pair_diff[k].append(score_list[k][i] - score_list[k][j])

    for i in range(len(metric_list)):
        for j in range(i):
            agree = 0
            for k in tqdm(range(cnt)):
                if pair_diff[i][k] * pair_diff[j][k] > 0:
                    agree += 1
            aggreement_matrix[i][j] = agree/cnt

    return aggreement_matrix


def render_agreement(matrix, dataset_name='none'):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i][j] = 1
            if j > i:
                matrix[i][j] = matrix[j][i]
            matrix[i][j] = round(matrix[i][j],2)

    x_label = [
        #"Prec", "ERR", "DCG@10", "RBP@Φ=0.5","RBP@Φ=0.8", "INST@T=1", "INST@T=5",
               "ReDeM-Init", "ReDeM-Max", "ReDeM-End", "ReDeM-Avg", "ReDeM-PE"]
    y_label = x_label

    corr_map = np.array(matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_map, cmap= plt.get_cmap("Blues"))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label)))
    ax.set_yticks(np.arange(len(y_label)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_label)):
        for j in range(len(y_label)):
            text = ax.text(j, i, corr_map[i, j],
                ha="center", va="center", color="k", fontsize = 8)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("ratio", rotation=-90, va="bottom")
    ax.set_title("Agreement Rate of Each Measure Pair")

    fig.tight_layout()
    plt.savefig('output/'+dataset_name+'_agreement.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    render_curve()
    #render_agreement(TIANGONG_AGREEMENT, 'tiangong')
