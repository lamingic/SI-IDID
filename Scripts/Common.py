# 公共方法
import numpy as np
import random
import math


def normalize_belief(_d, copy=True):
    """

    @param _d:
    @param copy:
    @return: 单个belief
    """
    d = _d if not copy else np.copy(_d)
    d /= np.sum(d)
    return d


def get_beliefs(M, k):
    """
    初始化belief
    @param M: 个数
    @param k: 维度
    @return: belief
    """
    belief = dict()
    for m in range(0, M):
        belief[m] = normalize_belief(np.random.rand(1, k), copy=False)[0]
    return belief


def enter_evidences_gui(idid, evidences, path):
    """
    将各值添加到IDID中
    @param idid: 模型IDID
    @param evidences: 名称
    @param path: path
    @return: 无
    """
    [idid.dbn.net.clear_evidence(ei) for ei in evidences]
    for ei_index in range(0, len(evidences)):
        ei = evidences[ei_index]
        idid.dbn.net.set_evidence(ei, int(path[ei_index]))
    idid.dbn.net.update_beliefs()


def gen_pathes_weights(pathes, weights):
    """
    轮盘赌获取path和weight
    @param pathes: pathes
    @param weights: weights
    @return: 单个path和weight
    """
    choose = np.random.randint(low=0, high=len(weights), size=1)[0]
    # choose = russian_roulette(weights)
    pathes = [pathes[choose]]  # compared one
    weights = [weights[choose]]  # compared one
    return pathes, weights


def russian_roulette(weights):
    """
    轮盘赌算法
    @param weights:
    @return: 通过轮盘赌获得的weight最大的索引
    """
    import random
    sum_w = np.sum(weights)
    sum_ = 0
    u = random.random() * sum_w
    for pos in range(0, len(weights)):
        sum_ += weights[pos]
        if sum_ > u:
            break
    return pos


def get_path_pr(idid, path):
    """
    将path代入到idid获取每个path的概率
    @param idid: idid模型
    @param path: path
    @return: 概率
    """
    # 时间片个数
    t = int((len(path) + 1) / 2)
    p = 1.0
    for i in range(0, t):
        action_value = idid.dbn.net.get_node_value("OD" + str(t - i))[int(path[i * 2])]
        observation_value = 1.0
        if (i != (t - 1)):
            observation_value = idid.dbn.net.get_node_value("OO" + str(t - i - 1))[int(path[i * 2 + 1])]
        p = p * action_value * observation_value
    return p


def get_most_possible_pathes(idid, policy_path_weight_j):
    """
    获取每棵树的每个path的概率
    @param idid: idid模型
    @param policy_path_weight_j: tree的path weight
    @return: 每棵树的每个path的概率
    """
    # num_mod_init_did = self.domain_parameters.values['num_mod_init_did']
    num_mod_did = idid.DomainParameters.values['num_mod_did']
    policy_dict_j = idid.did.dbn.result.get("policy_dict")
    column = len(policy_dict_j.get(0))
    path_pr_list = np.zeros((len(policy_dict_j), column), dtype=np.float64)  # 存储每个path的概率

    for modj in range(0, len(policy_dict_j)):
        pathes = policy_dict_j.get(modj)

        for pj in range(0, len(pathes)):
            # 获得path
            path = pathes[pj]
            # 计算每个path的概率
            path_pr = get_path_pr(idid, path)
            # path_pr_list[modj][pj] = path_pr
            path_pr_list[modj][pj] = path_pr
            policy_path_weight_j[modj][pj] = path_pr_list[modj][pj]
    return path_pr_list, policy_path_weight_j


def getPathPrList(idid, domain, pathes_i, weights_i, belief):
    """
    获取每个tree的每个path的内部idid计算操作
    @param idid:
    @param domain:
    @param pathes_i:
    @param weights_i:
    @param pathes_j:
    @param weights_j:
    @param belief:
    @return:
    """
    num_idid = domain.values.get('num_mod_idid')
    # mod_weigth = np.zeros(domain.values.get("num_mod_init_did"))

    for pi in range(0, len(pathes_i)):
        path_i = pathes_i[pi]
        weight_i = weights_i[pi]
        ei = "S" + str(idid.dbn.horizon)
        idid.dbn.net.set_virtual_evidence(ei, belief)
        # idid.dbn.net.update_beliefs()

        idid.dbn.net.set_evidence("O" + str(idid.dbn.horizon),
                                    random.randint(0, len(idid.dbn.observation_list) - 1))
        # idid.dbn.net.update_beliefs()
        enter_evidences_gui(idid, idid.dbn.evidences, path_i)
        for i in range(0, num_idid):
            policy_path_weight_j = idid.did.dbn.result.get('policy_path_weight')
            # 获取初始模型中的每个path的概率
            path_pr_list, policy_path_weight_j = get_most_possible_pathes(idid, policy_path_weight_j)
        idid.dbn.net.clear_all_evidence()

    return path_pr_list


def normalization(data_list):
    """
    数据标准化（Z-Score标准化）
    @param data_list:
    @return:
    """
    data_std = np.std(np.array(data_list))
    data_mean = np.mean(np.array(data_list))
    data_form = list()
    for data in data_list:
        data_form.append((data - data_mean) / data_std)
    return data_form


def normalization2(data_list):
    """
    最大最小值标准化
    @param data_list:
    @return:
    """
    data_min = np.min(np.array(data_list))
    data_max = np.max(np.array(data_list))
    data_form = list()
    for data in data_list:
        data_form.append((data - data_min) / (data_max - data_min))
    return data_form


def get_entropy(avg_path_pr_list):
    """
    根据每个tree的path的概率求信息熵并标准化
    @param avg_path_pr_list:
    @return: 每课tree的信息熵
    """
    # 1. 求信息熵
    xinxishang_list = list()

    for i in avg_path_pr_list:
        entropy = 0
        for j in i:
            if j > 0:
                entropy += j * math.log2(j)
        xinxishang_list.append(-entropy)
    return xinxishang_list


def norm_entropy(entropy_list):
    """
    标准化信息熵
    @param entropy_list:
    @return:
    """
    return normalization2(entropy_list)


def cal_path_pr(idid):
    """
    根据idid计算每个模型的pr
    @param idid: idid模型
    @return: pr
    """
    domain = idid.DomainParameters
    # 获取每个policy tree的path的概率
    # 1 获取i的path和权重
    N = 500
    belief = get_beliefs(N, domain.DBNS['IDID'].num_ss)
    policy_dict_j = idid.did.dbn.result.get('policy_dict')

    N_path_pr_list = np.zeros((N, domain.values['num_mod_did'], len(policy_dict_j.get(0))))

    # 100次循环后的结果（10个tree， 每个tree 4个path）
    for i in range(0, N):
        pathes_i = idid.dbn.result.get("policy_dict")
        weights_i = idid.dbn.result.get("policy_path_weight")
        pathes_i, weights_i = gen_pathes_weights(pathes_i.get(0), weights_i.get(0))

        # 交互的时候用的到
        # pathes_j = idid.did.dbn.result.get("policy_dict")
        # weights_j = idid.did.dbn.result.get("policy_path_weight")
        # pathes_j, weights_j = gen_pathes_weights(pathes_j.get(0), weights_j.get(0))

        path_pr_list = getPathPrList(idid, domain, pathes_i, weights_i, belief[i])
        for a in range(0, len(path_pr_list)):
            for b in range(0, len(path_pr_list[0])):
                N_path_pr_list[i][a][b] = path_pr_list[a][b]
    # 取平均后的结果
    avg_path_pr_list = np.mean(N_path_pr_list, axis=0)
    return avg_path_pr_list


def get_max_pro_path_index(list):
    """
    根据path的概率得到概率Topk的path的索引
    @param avg_path_pr_list:
    @return:
    """
    max = 0
    index_i = 0
    index_j = 0
    weight = []
    index = np.zeros((3, 2), dtype=np.int32)
    for k in range(0, 3):
        for i in range(0, len(list)):
            for j in range(0, len(list[i])):
                if (list[i][j] > max):
                    max = list[i][j]
                    index_i = i
                    index_j = j
        index[k][0] = index_i
        index[k][1] = index_j
        weight.append(max)
        list[index_i][index_j] = 0
        max = 0
    return index, np.array(weight)


def get_most_path(policy_dicts, index, weight_m):
    """
    获取topk path概率的path和weights
    @param policy_dicts:
    @param index:
    @param weight_m:
    @return:
    """
    most_path_list_m = list()
    w_m = weight_m / np.sum(weight_m)
    for i in index:
        most_path_list_m.append(policy_dicts.get(i[0])[i[1]])
    return most_path_list_m, w_m


def cal_distance(idid, policy_dicts, most_path_list_m, w_m):
    a_list = idid.did.dbn.action_list
    dis = np.zeros(len(policy_dicts))
    for i in range(len(policy_dicts)):
        pathes = policy_dicts.get(i)
        dis_ = distance(pathes, most_path_list_m, w_m, a_list)
        dis[i] = dis_
    return dis


def distance(pathes, most_possible_path_list, path_w, a_list):
    """
    最可能的path和每个模型之间的相似性
    @param pathes:
    @param most_possible_path_list:
    @param path_w:
    @param a_list:
    @return:
    """
    t = int((len(pathes[0]) + 1) / 2)
    cpt_1 = np.zeros((t, len(a_list)), dtype=np.float64)
    cpt_2 = np.zeros((t, len(a_list)), dtype=np.float64)
    # 1.根据候选模型, 求解在候选模型下各个属性的概率值
    for i in range(0, t):
        for j in range(0, len(pathes)):
            for a in range(0, len(a_list)):
                if (pathes[j][i * 2] == a):
                    cpt_1[i][a] += 1
    for i in range(0, len(cpt_1)):
        for j in range(0, len(cpt_1[0])):
            cpt_1[i][j] = cpt_1[i][j] / len(pathes)

    # 2. 根据求出的最可能的path， 求解各个属性的概率值
    for i in range(0, t):
        for j in range(0, len(most_possible_path_list)):
            for a in range(0, len(a_list)):
                if (most_possible_path_list[j][i * 2] == a):
                    cpt_2[i][a] += 1 * path_w[j]
    # d = 0
    # for i in range(0, t):
    #     for j in range(0, len(a_list)):
    #         d += np.abs(cpt_1[i][j] - cpt_2[i][j])
    # print(d)
    d = np.sum(np.abs(cpt_1 - cpt_2))
    return d


def cal_entropy(avg_path_pr_list):
    """
    计算信息熵
    @param avg_path_pr_list:
    @return:
    """
    # 求信息熵
    entropy_list = get_entropy(avg_path_pr_list)
    return entropy_list


def get_idid(idid, mod_j_path, mod_j_path_weight):
    """
    重新求解IDID并计算信息熵
    @param idid:
    @param mod_j_path:
    @param mod_j_path_weight:
    @return: 每个模型的信息熵
    """
    domain = idid.DomainParameters
    from IDID import IDID
    idid = IDID(idid.DomainParameters, idid.ModelParameters, idid.scr_message)

    idid.next_step(start=True)
    idid.dbn.extend(domain.values['horizon_size'], idid.step)
    idid.next_step(start=True, end=True)
    idid.dbn.generate_evidence(idid.step)
    idid.next_step(start=True, end=True)

    idid.did.next_step(start=True)
    idid.did.dbn.extend(domain.values['horizon_size'], idid.did.step)
    idid.did.next_step(start=True, end=True)
    idid.did.dbn.generate_evidence(idid.did.step)
    idid.did.next_step(start=True, end=True)

    idid.did.dbn.result["policy_dict"] = mod_j_path
    idid.did.dbn.result["policy_path_weight"] = mod_j_path_weight
    idid.did.dbn.result["policy_tree"].policy_dict = mod_j_path
    idid.did.next_step(start=True, end=True)
    # idid.did.extend(xinxi_IDID=None)
    idid.did.next_step(start=True, end=True)
    idid.did.dbn.result.get("policy_tree").gen_policy_trees_memorysaved()

    idid.did.dbn.result.get('policy_tree').set_name(
        idid.did.DomainParameters.Tostring() + '-' + idid.did.ModelParameters.Name + '-' + str(
            idid.did.step) + '-The Merged Policy Tree of ' + idid.did.type)
    idid.did.next_step(end=True)

    idid.next_step(start=True, end=True)
    idid.dbn.expa_policy_tree = idid.did.dbn.result.get("policy_tree")
    idid.dbn.expansion(idid.step)
    idid.next_step(start=True, end=True)
    # 求解IDID
    idid.solve_mod()
    idid.next_step(start=False, end=True)
    idid.expansion(expansion_flag=True)
    idid.next_step(start=True, end=False)
    idid.extend(public_idid=None)
    idid.next_step(start=True, end=True)
    idid.dbn.result.get("policy_tree").gen_policy_trees_memorysaved()
    idid.dbn.result.get('policy_tree').set_name(
        idid.DomainParameters.Tostring() + '-' + idid.ModelParameters.Name + '-' + str(idid.step) +
        '-The Merged Policy Tree of ' + idid.type)
    idid.next_step(end=True)
    idid.expansion(expansion_flag=False, policy_tree=idid.dbn.result.get('policy_tree'))
    return idid


def get_diversity(pops):
    """
    获取fitness
    @param pops:
    @return:
    """
    diversity = list()
    for pop in pops:
        diversity.append(pop.get("fitness"))
    return diversity


def norm_diversity(diversity_list):
    """
    标准化多样性
    @param pops:
    @return:
    """
    return normalization2(diversity_list)


def cal_fitness(norm_diversity, norm_entropy):
    """
    加权计算适应度函数
    @param norm_diversity: 多样性
    @param norm_entropy: 信息熵
    @return: 适应度值
    """
    nums = len(norm_diversity)
    fitness = list()
    for i in range(nums):
        fitness.append(norm_entropy[i] + norm_diversity[i])
    return fitness


def get_pbest(pop):
    """
    根据种群选择本次迭代最好粒子
    @param pop:
    @return:
    """
    # min_value = float('-inf')
    min_value = -100.0
    max_pop = None
    for p in pop:
        if p.get('fitness') > min_value:
            min_value = p.get('fitness')
            max_pop = p
    from GA import Gene
    g = Gene(data=[])
    g.data = max_pop.get('Gene').data
    d = dict()
    d['Gene'] = g
    d['fitness'] = max_pop.get('fitness')
    d['velocity'] = max_pop.get('velocity')
    return d


def get_dict(value):
    """
    构建字典
    @param value:
    @return:
    """
    dict_ = dict()
    for v in range(len(value)):
        dict_[v] = value[v]
    return dict_

def LCCS(A, B):
    """
    计算最长连续公共子序列
    @param A:
    @param B:
    @return:
    """
    max_length = 0
    max_index_A = 0
    max_index_B = 0

    length_A = len(A)
    length_B = len(B)

    dp = [[0] * (length_B + 1) for _ in range(length_A + 1)]

    for i in range(1, length_A + 1):
        for j in range(1, length_B + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    max_index_A = i
                    max_index_B = j

    start_index_A = max_index_A - max_length
    end_index_A = max_index_A - 1
    start_index_B = max_index_B - max_length
    end_index_B = max_index_B - 1

    return (start_index_A, end_index_A), (start_index_B, end_index_B)


def cal_cross(A, B):
    """
    交叉操作
    @param A:
    @param B:
    @return:
    """
    index_range_A, index_range_B = LCCS(A, B)
    d_1 = min(index_range_A[0], index_range_B[0])
    d_2 = max(index_range_A[1], index_range_B[1])

    C = [B[i] if i >= d_1 and i <= d_2 else A[i] for i in range(len(A))]
    return C