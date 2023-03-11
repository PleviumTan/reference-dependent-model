from config import *
import abc
import math


def f(i, k):
    if i > 0:
        return 1
    else:
        return 2 / (1 + math.exp(-k*i))


class Metric(metaclass=abc.ABCMeta):
    C = []
    W = []
    L = []
    V = []
    U = []
    estimate_depth = -1
    args = {}
    name = ''

    @abc.abstractmethod
    def read_serp(self, result_list):
        pass

    def get_etg(self):
        # ETG: expected total gain
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        for i in range(0, depth):
            res += L[i] * sum(U[:i+1])
        return res

    def get_erg(self):
        # ERG: expected rate of gain
        U = self.U
        V = self.V
        L = self.L
        depth = len(V)
        v_plus = sum(V[:depth])
        res = 0
        for i in range(0, depth):
            res += (L[i] * sum(U[:i+1])) / v_plus
        return res

    def get_avg(self):
        # AVG: average gain
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        for i in range(0, depth):
            res += L[i] * (sum(U[:i+1]) / (i+1))
        return res

    def get_max(self):
        # MAX: maximum gain
        U = self.U
        L = self.L
        depth = len(U)
        max_gain = -1
        res = 0
        for i in range(0, depth):
            if U[i] > max_gain:
                max_gain = U[i]
            res += L[i] * max_gain
        return res

    def get_fin(self):
        # FIN: Final gain
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        for i in range(0, depth):
            res += L[i] * U[i]
        return res

    def get_fg(self, dt=0.8):
        # FG: Forget (Recency Effect)
        # dt: Delta, the factor for forgetting
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        for i in range(0, depth):
            res = res * dt + L[i] * U[i]
        return res

    def get_pe(self, bt=0.5):
        # PE: Peak End Effect
        # bt: Beta, the factor for the weight of peak
        U = self.U
        L = self.L
        depth = len(U)
        max_gain = -1
        res = 0
        for i in range(0, depth):
            if U[i] > max_gain:
                max_gain = U[i]
            res = L[i] * (bt * max_gain + (1-bt) * U[i])
        return res

    def get_err(self):
        # If Metric is RR then this will become ERR
        L = self.L
        depth = len(L)
        res = 0
        for i in range(0, depth):
            res += L[i] * 1/(i + 1)
        return res

    def get_am(self, kp=10):
        # AM: Anchoring Metric
        # kp: Kappa, the factor for how fast the anchoring effect grows as the document quality increases
        U = self.U
        L = self.L
        depth = len(U)
        pre_rel = 0
        total_gain = 0
        res = 0
        for i in range(0, depth):
            if i:
                x = (pre_rel - 0.5)/0.5
                a = 1 / (1 + math.exp(-kp * x))
            else:
                a = 0
            gain = a * pre_rel + (1-a) * U[i]
            total_gain += gain
            res += L[i] * total_gain
            pre_rel = U[i]
        return res

    '''
    def get_ra_static(self, kp=2):
        # RA: Risk Aversion
        # kp: Kappa, the factor for risk aversion
        # Reference Point: The First
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        total_gain = 0
        ref = U[0]
        for i in range(0, depth):
            dt_r = U[i] - ref
            #gain = f(dt_r, kp) * U[i]
            #total_gain += gain
            total_gain += U[i]
            res += L[i] * (f(dt_r, kp) * total_gain)
        return res

    def get_ra_dynamic_1(self, kp=2):
        # RA: Risk Aversion
        # kp: Kappa, the factor for risk aversion
        # Reference Point: Average Relevance
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        total_gain = 0
        for i in range(0, depth):
            # Average relevance
            ref = sum(U[:i+1])/(i+1)
            dt_r = U[i] - ref
            #gain = f(dt_r, kp) * U[i]
            #total_gain += gain
            total_gain += U[i]
            res += L[i] * (f(dt_r, kp) * total_gain)
        return res

    def get_ra_dynamic_2(self, kp=2):
        # RA: Risk Aversion
        # kp: Kappa, the factor for risk aversion
        # Reference Point: The Best
        U = self.U
        L = self.L
        depth = len(U)
        res = 0
        total_gain = 0
        ref = U[0]
        for i in range(0, depth):
            # Average relevance
            dt_r = U[i] - ref
            #gain = f(dt_r, kp) * U[i]
            #total_gain += gain
            total_gain += U[i]
            res += L[i] * (f(dt_r, kp) * total_gain)
            if U[i] > ref:
                ref = U[i]
        return res
    '''

    def compute_v_l(self):
        C = self.C
        V = []
        L = []
        v = 1
        for c in C:
            V.append(v)
            v *= c
        self.V = V
        for i in range(0, len(V)):
            l = V[i] * (1 - C[i])
            L.append(l)
        self.L = L

    def compute_weight(self):
        tc = 1
        d = 0
        W = []
        for c in self.C:
            d += tc
            tc *= c
        tc = 1
        for c in self.C:
            W.append(tc/d)
            tc *= c
        self.W = W

    def compute_ed(self):
        ed = 0
        for rank in range(0, len(self.L)):
            ed += (rank+1) * self.L[rank]
        self.estimate_depth = ed

    def init(self):
        self.C = []
        self.W = []
        self.L = []
        self.V = []
        self.U = []
        self.args = {}
        self.name = ''


class DCG(Metric):
    def __init__(self, b=2):
        self.init()
        if b <= 1:
            raise ValueError
        self.args['b'] = b
        self.name = 'DCG-b='+str(b)

    def read_serp(self, result_list):
        rank = 0
        b = self.args['b']
        C = []
        U = []

        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                #c = (1 + math.log(rank + 1, b))/(1 + math.log(rank + 2, b))
                c = math.log(rank+2)/math.log(rank+3)
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RBP(Metric):
    def __init__(self, p):
        self.init()
        if p <= 0 or p > 1:
            raise ValueError('The parameter p must be in range of (0,1]')
        self.args['p'] = p
        self.name = 'RBP-p='+str(p)

    def read_serp(self, result_list):
        rank = 0
        p = self.args['p']
        C = []
        U = []
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # User will definitely stop at the truncation rank

            c = p
            C.append(c)
            U.append(cur_rel)
            rank += 1
        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RR(Metric):
    def __init__(self):
        self.init()
        self.name = 'ERR'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # Original?
            #ri = (pow(2, cur_rel*3) - 1)/pow(2, 3)
            ri = cur_rel
            # Binarize
            #if not cur_rel:
            #    ri = 0
            #else:
            #    ri = 1
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = 1 - ri
            C.append(c)
            U.append(cur_rel)
            rank += 1
        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class Prec(Metric):
    def __init__(self):
        self.init()
        self.name = 'Precision'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = 1
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class INSQ(Metric):
    def __init__(self, T):
        self.init()
        if T < 1:
            raise ValueError
        self.args['T'] = T
        self.name = 'INSQ-T='+str(T)

    def read_serp(self, result_list):
        rank = 0
        T = self.args['T']
        C = []
        U = []

        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = ((rank + 2*T)/(1 + rank + 2*T))**2
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class INST(Metric):
    def __init__(self, T):
        self.init()
        if T < 1:
            raise ValueError
        self.args['T'] = T
        self.name = 'INST-T='+str(T)

    def read_serp(self, result_list):
        rank = 0
        T = self.args['T']
        C = []
        U = []
        tr = 0.0

        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            tr += cur_rel
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = ((rank + 2*T - tr)/(1 + rank + 2*T - tr))**2
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class AP(Metric):
    def __init__(self):
        self.init()
        self.name = 'AP'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []

        sum_rj = 0
        pre_sum = 0
        for j in range(len(result_list)):
            sum_rj += float(result_list[j] / (j + 1))

        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            # WIP
            cur_sum = pre_sum + cur_rel / (rank + 1)
            if sum_rj - pre_sum:
                c = (sum_rj - cur_sum) / (sum_rj - pre_sum)
            else:
                c = 0

            C.append(c)
            U.append(cur_rel)
            rank += 1
            pre_sum = cur_sum

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RAStatic1(Metric):

    def __init__(self):
        self.init()
        self.name = 'RA_Static_1'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        total_rel = 0
        cur_k = -1
        max_k = -1
        ref = 0
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])
            if not rank:
                #ref: i = 1
                ref = cur_rel
            total_rel += cur_rel
            cur_k = total_rel/(rank+1)
            if cur_k > max_k:
                max_k = cur_k
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                c = (2 + rank - cur_rel)/(3 + rank - cur_rel + ref)
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RADynamic1(Metric):
    def __init__(self):
        self.init()
        self.name = 'RA_Dynamic_1_max'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        total_rel = 0
        cur_k = -1
        max_k = -1
        ref = 0
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])

            total_rel += cur_rel
            cur_k = total_rel/(rank+1)
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                # c = math.pow(((rank+1)/(rank+2)), 1-lm) * math.pow(((2 - cur_rel)/(2 - cur_rel + ref)), lm)
                c = ((2 + rank - cur_rel)/(3 + rank - cur_rel + ref))
            # reference: max
            if cur_rel > ref:
                ref = cur_rel
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RADynamic2(Metric):

    def __init__(self):
        self.init()
        self.name = 'RA_Dynamic_2_end'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        total_rel = 0
        cur_k = -1
        max_k = -1
        ref = 0
        while rank < D_VIEW:
            if rank >= D_REL:
                cur_rel = 0.0
            else:
                cur_rel = float(result_list[rank])

            total_rel += cur_rel
            cur_k = total_rel/(rank+1)
            # User will definitely stop at the truncation rank
            if rank == D_VIEW - 1:
                c = 0
            else:
                # c = math.pow(((rank+1)/(rank+2)), 1-lm) * math.pow(((2 - cur_rel)/(2 - cur_rel + ref)), lm)
                c = ((2 + rank - cur_rel)/(3 + rank - cur_rel + ref))
            # reference: end
            ref = cur_rel
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RADynamic3(Metric):

    def __init__(self):
        self.init()
        self.name = 'RA_Dynamic_3_peak_end_0.5'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        total_rel = 0
        max_rel = 0
        end = 0
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
                ref = 0.5 * max_rel + 0.5 * end
                c = ((2 + rank - cur_rel)/(3 + rank - cur_rel + ref))
            # reference: max
            if cur_rel > max_rel:
                max_rel = cur_rel
            # reference: end
            end = cur_rel
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()


class RADynamic4(Metric):

    def __init__(self):
        self.init()
        self.name = 'RA_Dynamic_4_avg'

    def read_serp(self, result_list):
        rank = 0
        C = []
        U = []
        total_rel = 0
        ref = 0
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
                c = ((2 + rank - cur_rel)/(3 + rank - cur_rel + ref))

            ref = total_rel/(rank+1)
            C.append(c)
            U.append(cur_rel)
            rank += 1

        self.C = C
        self.U = U
        self.compute_weight()
        self.compute_v_l()
        self.compute_ed()
