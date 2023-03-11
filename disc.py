import re
import os
import pandas as pd
from config import *
from utils import *
from tqdm import tqdm

'''
Parse Qrel Files
'''
def parse_row(row):
    el = row.split(' ')
    el[2] = int(re.findall('[0-9]+', el[2])[0])
    return el

def disc_power():

    # Read test collection
    for tc in TEST_COLLECTION:
        tcdir = DISC_DIR + '/' + tc
        fl = os.listdir(tcdir)

        qrel_file = list(filter(lambda f: re.match(r".*\.qrels", f) , fl))[0]
        rundir = tcdir + '/Runs'

        # Parse qrel file
        with open(tcdir + '/' + qrel_file, 'r+', encoding='utf-8') as f:
            qrel_records = [parse_row(x) for x in f.readlines()]

        #print(qrel_records[0])
        df_qrel = pd.DataFrame(qrel_records, columns=['top_id','doc_id', 'rel'])
        df_qrel[['top_id']] = df_qrel[['top_id']].astype(str)
        max_rel = df_qrel['rel'].max()
        df_qrel[['rel']] = df_qrel[['rel']].apply(lambda x: x/max_rel)

        qrel_map = {}
        for idx, rec in df_qrel.iterrows():
            qrel_map[rec['top_id'] + rec['doc_id']] = rec['rel']

        sys_topic_map = {}
        for mn in METRIC_LIST:
            if mn not in sys_topic_map:
                sys_topic_map[mn] = []

        # Read run data
        rl = os.listdir(rundir)
        for sys in tqdm(rl):
            with open(rundir + '/' + sys, 'r+', encoding='utf-8') as f:
                run_records = []
                serp = []
                cur_top = ''
                cnt = 999
                for row in f.readlines():
                    # Skip the first row
                    if re.findall('<SYSDESC>', row):
                        #print(sys)
                        continue
                    parsed_row = re.split('\s', row)
                    #print(el)
                    if parsed_row[0] != cur_top and cnt >= 10:
                        #print(parsed_row[0])
                        if len(serp):
                            run_records.append(serp)
                        serp = []
                        cur_top = parsed_row[0]
                        cnt = 0
                    elif cnt >= 10:
                        continue

                    if parsed_row[0] + parsed_row[2] in qrel_map:
                        serp.append(qrel_map[parsed_row[0] + parsed_row[2]])
                    else:
                        serp.append(0)

                    # Too slow!!!
                    '''
                    rel_record = df_qrel[(df_qrel['top_id'] == parsed_row[0]) & (df_qrel['doc_id'] == parsed_row[2])]

                    if not rel_record.empty:
                        serp.append(rel_record['rel'].item())
                    else:
                        serp.append(0)
                    '''

                    cnt += 1

                # Don't forget the last topic
                run_records.append(serp)
                #print(run_records)
                for mn in METRIC_LIST:
                    topic_scores = []
                    if mn in DEFAULT_PARAMETER:
                        metric = get_my_metric(mn, DEFAULT_PARAMETER[mn])
                    else:
                        metric = get_my_metric(mn, -1)
                    for serp in run_records:
                        metric.read_serp(serp)
                        if DEFAULT_A_MODE[mn] == 'ERG':
                            topic_scores.append(metric.get_erg())
                        elif DEFAULT_A_MODE[mn] == 'ETG':
                            topic_scores.append(metric.get_etg())
                        elif DEFAULT_A_MODE[mn] == 'ERR':
                            topic_scores.append(metric.get_err())
                    #print(mn)
                    #print(topic_scores)
                    sys_topic_map[mn].append(topic_scores)

        for mn in sys_topic_map:
            mx = sys_topic_map[mn]
            min_len = len(mx[0])
            for row in mx:
                if len(row) < min_len:
                    min_len = len(row)

            rot_mx = []
            for i in range(min_len):
                row = []
                for j in range(len(mx)):
                    row.append(mx[j][i])
                rot_mx.append(row)

            df_mx = pd.DataFrame(rot_mx)
            df_mx.to_csv( mn + '.' + tc +  '.topicbyrun', sep=' ', index=False, header=False)


if __name__ == '__main__':
    disc_power()







