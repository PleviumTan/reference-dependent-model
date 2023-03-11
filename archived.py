'''
Archive
'''


'''
Parse TianGong SS-FSD
'''


def tiangong_fsd_parse(data_dir):
    res = []
    cnt = 0
    map_list = ['difficulty', 'urgency', 'atmosphere', 'trigger', 'specificity']
    print("Reading data...")
    with open(data_dir) as f:
        log = json.load(f)
    print("TianGong Qref Successfully Loaded")
    for session in log:
        for query in session['queries']:
            cnt += 1
            query['parse_id'] = str(session['user_id']) + '-' + str(session['session_id']) + str(query['query_id'])
            query['SERPs'][0]['top10_usefulness'] = list(
                map(lambda i: i / MAX_REL_QREF, query['SERPs'][0]['top10_usefulness']))

            for key in map_list:
                query[key] = session[key]
            res.append(query)
    print("TianGong Qref Successfully Parsed.\n")
    print("Total: " + str(cnt) + " results\n")
    return res


class Cognitive_Cost(Metric):

    def __init__(self, lm):
        self.init()
        if lm < 0:
            raise ValueError('The parameter lambda must be >= 0')
        self.args['lm'] = lm
        self.name = 'RA_Dynamic4-lm='+str(lm)

    def read_serp(self, result_list):
        rank = 0
        lm = self.args['lm']
        C = []
        U = []
        total_rel = 0
        cur_k = 0
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            total_rel += cur_rel
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = 1/(1 + lm * (1 + cur_rel)/(rank+2))
            cur_k = total_rel/(rank+1)
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RAStatic_CD(Metric):

    def __init__(self, lm):
        self.init()
        if lm < 0:
            raise ValueError('The parameter lambda must be >= 0')
        self.args['lm'] = lm
        self.name = 'RA_Static_2-lm='+str(lm)

    def read_serp(self, result_list):
        rank = 0
        lm = self.args['lm']
        C = []
        U = []
        total_rel = 0
        cur_k = 0
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            if not rank:
                ref = cur_rel #ref: i = 1
            total_rel += cur_rel
            cur_k = total_rel/(rank+1)
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = math.pow((rank+1)/(rank+2), 1-lm) * math.pow((2 - cur_rel)/(2 - cur_rel + ref), lm)
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()



