import math

from config import *
from metrics import *
import json
import scipy
from scipy.stats import norm
import numpy as np
try:
  import xml.etree.cElementTree as et
except ImportError:
  import xml.etree.ElementTree as et
import pandas as pd
import re


'''
Fisher's CI for Kendall's tau
'''
def get_CI(tv, sz, sig = 0.05):
    zr = math.log((1 + tv)/(1 - tv))/2
    mg = norm.ppf(1 - sig/2) * math.sqrt(0.437/(sz - 4))
    zl = zr - mg
    zu = zr + mg

    rl = (math.exp(2 * zl) - 1)/(math.exp(2 * zl) + 1)
    ru = (math.exp(2 * zu) - 1)/(math.exp(2 * zu) + 1)

    return (rl, ru)


'''
Parse THUIR1
'''


def thuir1_parse(data_dir, score_file, rel_file):
    parsed_log = []

    with open(score_file, 'r+', encoding='utf-8') as f:
        ss = [i[:-1].split('\t') for i in f.readlines()]
    with open(rel_file, 'r+', encoding='utf-8') as f:
        rels = [i[:-1].split('\t') for i in f.readlines()]

    cnt_err = 0
    for rec in ss:
        sat = -1
        session_id = rec[0]
        serp_id = session_id.split('_')[1]
        cl = []  # click list
        # Parse log
        # print(session_id)
        try:
            with open(data_dir + session_id, 'r+', encoding='utf-8') as f:
                for i in f.readlines():
                    l = i[:-1].split('\t')
                    if l[2] == 'ACTION=SATISFY':
                        # Extract satisfaction score
                        sat = float(l[4][-1])
                    if l[2] == 'ACTION=CLICK':
                        # Extract distinct click record
                        try:
                            c = int(l[5][-1])
                            if c not in cl:
                                cl.append(c)
                        except:
                            pass
        except:
            cnt_err += 1
            continue

        nc = len(cl)
        if nc:
            dc = max(cl) + 1
        else:
            dc = 0
        query_result = []
        for rel in rels:
            if rel[0] == serp_id:
                for rank in range(0, 10):
                    query_result.append((float(rel[rank + 1])-1) / MAX_REL_THUIR1)
                    #query_result.append(float((pow(2, float(rel[rank + 1])-1)) - 1)/pow(2, MAX_REL_THUIR1))
        parsed_log.append({'id': session_id,
                           'SERPs': [{'top10_usefulness': query_result}],
                           'satisfaction': sat, 'nc': nc, 'dc': dc})

    print("Success: %s, Failure: %s" % (len(parsed_log), cnt_err))
    return parsed_log


'''
Parse THUIR2
'''
def thuir2_parse(data_dir, rel_file):
    myParser = et.XMLParser(encoding="utf-8")
    tree = et.parse(data_dir, parser=myParser)
    root = tree.getroot()
    cnt = 0
    cnt_ss = 0
    '''
    Read tsv data
    '''
    rel_data = pd.read_csv(rel_file, sep = '\t', header = 0, encoding='utf-8')

    '''
    Read XML data
    '''
    parsed_log = []
    for session in root.iter("session"):
        cnt_ss += 1
        session_id = session.get("num")
        query_id = ""
        for query in session.iter("interaction"):
            cnt += 1
            page_id = query.get("page_id")
            if page_id == "1":
                if query_id:
                    # Add record to log
                    parsed_log.append({"id":session_id+"-"+query_id, "query": query_text,
                           'SERPs': [{'top10_usefulness':query_result}], 'satisfaction': query_sat, "nc": nc, "dc": dc})
                query_id = query.get("num")
                query_text = query.find("query").text
                query_result = []
                query_sat = query.find("query_satisfaction").get("score")
                nc = 0 # number of clicks
                dc = 0 # deepest click
            current_result = query.find("results")
            current_click = query.find("clicked")
            if len(current_result) < 10:
                continue
            for result in current_result:
                doc_id = result.find("id").text
                doc_rank = result.get("rank")
                rel = rel_data[rel_data['docno'] == int(doc_id)]
                if len(rel):
                    doc_rel = rel.iloc[0, 3]
                else:
                    doc_rel = 0
                query_result.append(doc_rel/4)
            if current_click:
                for click in current_click:
                    nc += 1
                    dc = click.find('rank').text
        # Add record to log
        parsed_log.append({"id":session_id+"-"+query_id, "query": query_text,
                           'SERPs': [{'top10_usefulness':query_result}], 'satisfaction': query_sat, "nc": nc, "dc": dc})
    print("THUIR2 sucessfully parsed: "+ str(cnt_ss) + " sessions and "+str(cnt) + " queries.")
    return parsed_log
'''
Parse KDD-19
'''
def parse_kdd(log_file, rel_file):
        parsed_log = []
        rel_map = {}
        cnt = 0
        cnt_ss = 0

        # Parse Usefulness Score
        with open(rel_file,  'r+', encoding='utf-8') as f:
            tsvreader = pd.read_csv(f, sep='\t', header = 0)
            for row in tsvreader.iterrows():
                doc_id = str(row[1]['docno'])
                #rel_map[doc_id] = row[1]['relevance']
                rel_map[doc_id] = row[1]['usefulness_annotation']

        # Parse Session Log
        myParser = et.XMLParser(encoding="utf-8")
        tree = et.parse(log_file, parser=myParser)
        root = tree.getroot()
        for session in root.iter("session"):
            cnt_ss += 1
            user_id = str(session.get("userid"))
            topic = session.find('topic')
            topic_id = str(topic.get('num'))
            for query in session.iter("interaction"):
                query_id = str(int(query.get('num')) - 1)
                results = query.find('results').findall("result")
                result_list = []
                for doc in results:
                    doc_id = doc.find("id").text

                    try:
                        result_list.append(min((rel_map[doc_id] - 1)/MAX_USEFUL_KDD, 1))
                        #result_list.append(min(float((pow(2, float(rel_map[doc_id])-1)) - 1)/pow(2, MAX_REL_KDD), 1))
                    except:
                        result_list.append(0)
                if not len(result_list):
                    continue
                cnt += 1
                qsat = int(query.find('query_satisfaction').get('score'))
                parsed_log.append({'id': user_id + '-' + topic_id + '-' + query_id,
                                   'satisfaction': qsat, 'SERPs': [{'top10_usefulness':result_list}]})

        print("KDD19 sucessfully parsed. Total: " + str(cnt_ss) + " sessions and " + str(cnt) + " queries.")
        return parsed_log


'''
Parse TianGong Qref
'''


def tiangong_qref_parse(data_dir):
    res = []
    cnt = 0
    cnt_ss = 0
    click_map = {}
    feature_map = {}
    map_list = ['difficulty', 'urgency', 'expertise', 'trigger', 'specificity']
    print("Reading data...")
    with open(data_dir) as f:
        log = json.load(f)
    print("TianGong Qref Successfully Loaded")
    for session in log:
        #if len(session['queries']) > 10:
        #    continue
        cnt_ss += 1
        for query in session['queries']:
            query['parse_id'] = str(session['user_id']) + '-' + str(session['session_id']) + str(query['query_id'])
            query['SERPs'][0]['top10_usefulness'] = list(
                map(lambda i: float(i/MAX_REL_QREF), query['SERPs'][0]['top10_usefulness']))
                #map(lambda i: float((pow(2, i) - 1)/pow(2, MAX_REL_QREF)), query['SERPs'][0]['top10_usefulness']))
            cl = list(map(lambda i: i['id'], query['SERPs'][0]["clicked_results"]))
            if len(cl):
                # Drop SERPs where click depth > 10
                '''
                if max(cl) > 10:
                    continue
                '''
                query['SERPs'][0]['dc'] = max(cl) - 1
                if max(cl) == 0:
                    query['SERPs'][0]['dc'] = 0
                if max(cl) not in click_map:
                    click_map[max(cl)] = 0
                click_map[max(cl)] += 1
            else:
                query['SERPs'][0]['dc'] = -1

            for key in map_list:
                if key not in feature_map:
                    feature_map[key] = {}
                val = session[key]
                if val not in feature_map[key]:
                    feature_map[key][val] = 0
                feature_map[key][val] += 1
                query[key] = val
            cnt += 1
            res.append(query)
    print("TianGong Qref Successfully Parsed.")
    print("Total: " + str(cnt_ss) + " sessions and " + str(cnt) + " results")
    click_map = sorted(click_map.items())
    print(click_map)
    for key in feature_map:
        feature_map[key] = sorted(feature_map[key].items())
        print(">>>>>>"+key+":")
        print(feature_map[key])
    return res

'''
Parse TianGong FSD
'''


def tiangong_fsd_parse(data_dir):
    res = []
    cnt = 0
    cnt_ss = 0
    ss_len = [0 for i in range(100)]
    print("Reading data...")
    with open(data_dir) as f:
        log = json.load(f)
    print("TianGong SS-FSD Successfully Loaded")
    for session in log:
        #if len(session['queries']) > 10:
        #    continue
        cnt_ss += 1
        pre_kw = ''
        for i in range(len(session['queries'])):
            query = session['queries'][i]
            kw = query['query_string']

            query['parse_id'] = str(session['user_id']) + '-' + str(session['session_id']) + str(query['query_id'])
            query['SERPs'][0]['top10_usefulness'] = [(i['usefulness'])/MAX_USEFUL_FSD for i in query['SERPs'][0]['results']]

            clicked_res = []
            for item in query['SERPs'][0]['results']:
                if item['clicked']:
                    clicked_res.append({
                        "id": item["rank"],
                        "timestamp": item["click_timestamp"]
                    })
            query['SERPs'][0]['clicked_results'] = clicked_res
            if len(query['SERPs'][0]['top10_usefulness']) < 10:
                # print(len(query['SERPs'][0]['top10_usefulness']))
                for i in range(10 - len(query['SERPs'][0]['top10_usefulness'])):
                    query['SERPs'][0]['top10_usefulness'].append(0)
            query['SERPs'][0]['start_timestamp'] = query['start_timestamp']
            if not len(query['SERPs'][0]['time_intervals']):
                if not len(query['SERPs'][0]['mouse_moves']):
                    query['SERPs'][0]['end_timestamp'] = query['start_timestamp']
                    #print(query)
                else:
                    query['SERPs'][0]['end_timestamp'] = query['SERPs'][0]['mouse_moves'][-1]['Et']
            else:
                query['SERPs'][0]['end_timestamp'] = query['SERPs'][0]['time_intervals'][-1]['outT']

            cnt += 1
            pre_kw = kw
            #if reform_type == REFORM_TYPE_MAP_CHEN['ADD'] and query['relation'] == 1:
            #    print("This: "+ session['queries'][i]['query_string'] + " Pre: " + session['queries'][i-1]['query_string'])
        res.append(query)
        ss_len[len(session['queries'])] += 1
    print("TianGong SS-FSD Successfully Parsed.")
    print("Total: " + str(cnt_ss) + " sessions and " + str(cnt) + " results")
    print(ss_len)

    return res


'''
Useful　Methods
'''

def get_my_metric(metric_name, parameter):
    if metric_name not in METRIC_LIST:
        raise ValueError("No such metric: " + metric_name)
    if metric_name == 'RR':
        return RR()
    elif metric_name == 'AP':
        return AP()
    elif metric_name == 'Prec':
        return Prec()
    elif metric_name == 'DCG':
        return DCG(parameter)
    elif metric_name == 'RBP':
        return RBP(parameter)
    elif metric_name == 'INSQ':
        return INSQ(parameter)
    elif metric_name == 'INST':
        return INST(parameter)
    elif metric_name == 'RA_D1':
        return RADynamic1()
    elif metric_name == 'RA_D2':
        return RADynamic2()
    elif metric_name == 'RA_D3':
        return RADynamic3()
    elif metric_name == 'RA_D4':
        return RADynamic4()
    elif metric_name == 'RA_S1':
        return RAStatic1()
    else:
        raise Exception("Metric Undefined:" + metric_name)



def compute_TSE(train_set, metric):
    # Total Sum of Error
    tse = 0

    for record in train_set:
        dc = record['SERPs'][0]['dc']
        metric.read_serp(record['SERPs'][0]['top10_usefulness'])
        tse += (dc + 1 - metric.estimate_depth) ** 2
    return tse


def compute_corr(train_set, metric, Y):
    X = []
    for record in train_set:
        metric.read_serp(record['SERPs'][0]['top10_usefulness'])
        X.append(metric.get_erg())
    res = scipy.stats.spearmanr(X, Y).correlation
    return res


def low_difficulty(record):
    return record['difficulty'] <= 1


def high_difficulty(record):
    return record['difficulty'] > 1


def low_urgency(record):
    return record['urgency'] <= 1


def high_urgency(record):
    return record['urgency'] > 1


def low_expertise(record):
    return record['expertise'] <= 1


def high_expertise(record):
    return record['expertise'] > 1

def mid_cost(record):
    return (high_difficulty(record) and low_urgency(record)) or (low_difficulty(record) and high_urgency(record))

def high_cost(record):
    return (high_difficulty(record) and high_urgency(record)) or (high_difficulty(record) and high_urgency(record))

def low_cost(record):
    return (low_difficulty(record) and low_urgency(record)) or (low_difficulty(record) and low_urgency(record))


'''
Train　Baseline
'''


def tune_metric_parameter(train_set, metric_name, p_1, p_2, step):
    if metric_name in METRIC_LIST and metric_name not in METRIC_PARAMETER_BOUND:
        return -1
    parameter_map = {}
    y = [q['satisfaction'] for q in train_set]

    for p in np.arange(p_1, p_2, step):
        # print("p:%2f \n" % p)
        p = round(p, 2)
        #print("%s parameter: %2f" % (metric_name, p))
        metric = get_my_metric(metric_name, p)
        #parameter_map[p] = compute_TSE(train_set, metric)
        parameter_map[p] = compute_corr(train_set, metric, y)
        #p_st = min(parameter_map, key=parameter_map.get)
        p_st = max(parameter_map, key=parameter_map.get)

    #metric = get_my_metric(metric_name, p_st)
    #print("%s Best parameter:%2f TSE/Corr:%4f" % (metric_name, p_st, parameter_map[p_st]))
    return p_st


'''
Get X
'''


def get_x(input_log, metric):
    x_map = {}
    for record in input_log:
        metric.read_serp(record['SERPs'][0]['top10_usefulness'])
        if metric.name == 'ERR':
            if 'ERR' not in x_map:
                x_map['ERR'] = []
            x_map['ERR'].append(metric.get_err())
        else:
            if 'ERG' not in x_map:
                x_map['ERG'] = []
            x_map['ERG'].append(metric.get_erg())

        if re.search(r'RA', metric.name):
            if 'ETG' not in x_map:
                x_map['ETG'] = []
            x_map['ETG'].append(metric.get_etg())
            if 'AVG' not in x_map:
                x_map['AVG'] = []
            x_map['AVG'].append(metric.get_avg())
            if 'MAX' not in x_map:
                x_map['MAX'] = []
            x_map['MAX'].append(metric.get_max())
            if 'END' not in x_map:
                x_map['END'] = []
            x_map['END'].append(metric.get_fin())
            if 'PE' not in x_map:
                x_map['PE'] = []
            x_map['PE'].append(metric.get_pe())

    return x_map
