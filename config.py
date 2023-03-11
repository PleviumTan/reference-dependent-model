D_VIEW = 10
D_REL = 10

MAX_REL_QREF = 3
MAX_REL_THUIR1 = 3
MAX_USEFUL_KDD = 3
MAX_USEFUL_FSD = 4

# THUIR1
THUIR1_DIR = "data/THUIR1/search_sessions/"
THUIR1_SCORE_FILE="data/THUIR1/metrics.txt"
THUIR1_REL_FILE = "data/THUIR1/relevance.txt"

#KDD19
KDD_LOG_FILE = "data/KDD19/search_logs.xml"
KDD_REL_FILE = "data/KDD19/relevance_annotation.tsv"
KDD_USEFUL_FILE = "data/KDD19/usefulness_annotation.tsv"

# THUIR2
THUIR2_DATA_FILE = "data/THUIR2/search_logs.xml"
THUIR2_REL_FILE = "data/THUIR2/relevance_annotation.tsv"

QREF_DIR = "data/tiangong-qref-new.json"

FSD_DIR = "data/TianGong-SS-FSD.json"

DISC_DIR = '../ECIR2021rawdata'

'''
Path structure:
/Runs
***.qrels
'''

TEST_COLLECTION = [
    'WWW3',
    #'STC2',
    #'TR18Core',
    #'TR19DL'
]

'''
QREF_REL_MAP = {
    0: 0,
    1: 1/3,
    2: 2/3,
    3: 1
}
'''

# Expected Total Gain, Expected Rate of Gain, Average, ERR, Maximum, End, Peak End, Recency, Risk Aversion, Anchoring
METRIC_A_MODE = ['ETG', 'ERG', 'ERR'
                 #'AVG', 'MAX', 'FIN', 'PE', 'FG', 'AM'
                 ]
BASELINE_LIST = ['Prec', 'DCG', 'RBP', 'RR', 'INST']
RM_LIST = ['RA_S1', 'RA_D1', 'RA_D2', 'RA_D3', 'RA_D4']

METRIC_LIST = BASELINE_LIST + RM_LIST

DEFAULT_PARAMETER = {'DCG': 2, 'RBP': 0.8, 'INSQ': 2.25, 'INST': 2.25}

DEFAULT_A_MODE = {
    'Prec': 'ERG',
    'DCG': 'ERG',
    'RBP': 'ERG',
    'RR': 'ERR',
    'INST': 'ERG',
    'AP': 'ERG',
    'RA_S1': 'ERG',
    'RA_D1': 'ERG',
    'RA_D2': 'ERG',
    'RA_D4': 'ERG',
    'RA_D3': 'ERG'
}

METRIC_PARAMETER_BOUND = {
    'DCG': [2.0, 5.01, 0.1],
    'RBP': [0.1, 0.91, 0.05],
    'INST': [1, 21, 1],
}
