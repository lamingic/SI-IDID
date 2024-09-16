# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins
import copy
import random
import math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
# self package
from PolicyTree import PolicyTree
from DBN import DBN
from namespace import Name
import Common


###########################################################
class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)


class MA(object):
    def __init__(self, solver, filepath, result, dstr, num_mod):
        self.domain = dstr
        self.solver = solver
        self.type = 'MA'
        print('Initializing ' + self.type)
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.filename = solver.get('parameters')
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.result = result
        self.step = solver.get('step')
        self.alg_name = ''

    def set_alg_name(self, name):
        self.alg_name = name

    def main(self):
        if os.path.isfile(self.filename):
            f = open(self.filename, 'r')
            contents = [x.strip() for x in f.readlines()]
            self.result['policy_dict'] = dict()
            modi = -1
            for i in range(0, len(contents)):
                line = contents[i].split(' ')
                if len(line) == 1:
                    if line[0] == '':
                        continue
                    pathes = list()
                    modi = modi + 1
                else:
                    path = np.array([int(e) for e in line])
                    pathes.append(path)
                self.result['policy_dict'][modi] = pathes
            num_mod = modi
            rewards = dict()
            prior_belief = dict()
            policy_path_weight = dict()
            for modi in range(0, num_mod):
                policy_dict = dict()
                policy_dict[1] = self.result['policy_dict'][modi]
                policy_tree = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
                    self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
                    'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list,
                                         self.dbn.observation_list)
                policy_tree.set_policy_dict(policy_dict)
                policy_tree.gen_policy_trees_memorysaved()
                # policy_tree.save_policytree(self.pnames.Save_filepath)
                self.dbn.expa_policy_tree = policy_tree
                self.dbn.expansion(self.step, expansion_flag=False)
                # if self.parameters.values.get('cover_mode'):
                #     rewards[modi], policy_path_weight[modi], prior_belief[modi] = self.dbn.get_reward(weight_off=True, modi=gi)
                # else:
                #     rewards[modi], policy_path_weight[modi], prior_belief[modi] = self.dbn.get_reward(weight_off=True)
                rewards[modi], policy_path_weight[modi], prior_belief[modi] = self.dbn.get_reward()
            self.result['reward'] = rewards
            self.result['policy_path_weight'] = policy_path_weight
            self.result['prior_belief'] = prior_belief
            self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
                'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list,
                                                    self.dbn.observation_list)
            self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
            self.result['policy_tree'].gen_policy_trees_memorysaved()
            # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)
            # self.dbn.result = self.result


class GA(object):
    def __init__(self, solver, filepath, result, dstr, num_mod, type=None):
        if type is None:
            self.type = 'GA'
        else:
            self.type = type
        # print('Initializing ' + self.type)
        self.plot_flag = False  # for GUI GEN EXE
        self.domain = dstr
        self.pnames = Name()
        self.solver = solver
        self.num_mod = num_mod
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.result = result
        self.check_identity_flag = True
        self.progressbar = True
        self.initialize()
        self.initialize_pop()
        self.alg_name = ''

    def set_alg_name(self, name):
        self.alg_name = name

    # initialize
    def initialize(self):
        self.genomes = dict()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()

    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path

    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path

    def fill_tree_template(self):
        # self.tree_template = np.zeros([self.get_num_path(), self.get_len_path()])
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                # pathes = self.result['policy_dict'].get(key)
                self.tree_template = self.result['policy_dict'].get(key)
                break
            # for i in range(0, self.get_num_path()):
            #     self.tree_template[i, :] = pathes[i]
        # print(self.tree_template)

    def initialize_other(self):
        pass

    def initialize_pop(self):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate()  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'id': 0}

    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()
        self.gen_weight()

    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)

    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for j in range(0, self.get_len_path(), 2):
            hi = int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def pathes_to_mat(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        for i in range(0, self.get_num_path()):
            mat[i, :] = [pathes[i][j] for j in range(0, self.get_len_path(), 2)]
        return mat

    def mat_to_genome(self, mat):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.get_num_observation(), self.dbn.horizon - cl - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(mat[rw, cl]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def gen_genome_level(self):
        '''
             a	o	a	o	a        a	a  a
             2	0	0	0	2        2  0  2
             2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0 =>[3 2 2 1 1 1 1]
             2	1	2	0	2        2  2  2
             2	1	2	1	0        2  2  0
             '''
        if self.genome_template == None:
            print('============errrrrr')
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.dbn.horizon - cl
            start = start + num
        self.genome_level = genome_level
        # print('-------------------------------')
        # print(genome_level)

    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.dbn.horizon - 1):
            step = np.power(self.dbn.num_os, cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.dbn.num_os)
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
        # print('-------------------------------')
        # print(self.genome_arc)

    def gen_weight(self):
        if not self.parameters.values.get('weight_mode'):
            pass
        w = 1 / self.dbn.horizon
        level = self.genome_level
        self.weight = np.array([w / (level.count(level[i])) for i in range(0, len(level))])
        # print(self.weight)

    def gen_pop(self):
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo))
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]

    def evaluate(self):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()

    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    ##
    # def gen_genome_subtree(self, gen):
    #     subtree = set()
    #     for gi in range(0, len(gen)):
    #         cl = self.genome_level[gi]
    #         if cl <= 1:
    #             break
    #         tree = list()
    #         tree.append(gen[gi])
    #         if not self.sub_genomes_total.__contains__(str(tree)):
    #             value = len(self.sub_genomes_total) + 1
    #             self.sub_genomes_total[str(tree)] = value
    #         else:
    #             value = self.sub_genomes_total.get(str(tree))
    #         subtree.add(value)
    #         parents = [gi]
    #         while cl > 0:
    #             children = list()
    #             for pa in parents:
    #                 for gj in range(gi, len(gen)):
    #                     if self.genome_arc[gj] == pa:
    #                         children.append(gj)
    #                         tree.append(gen[gj])
    #             if not self.sub_genomes_total.__contains__(str(tree)):
    #                 value = len(self.sub_genomes_total) + 1
    #                 self.sub_genomes_total[str(tree)] = value
    #             else:
    #                 value = self.sub_genomes_total.get(str(tree))
    #             subtree.add(value)
    #             parents = children
    #             # [parents.append(ch) for ch in  children]
    #             cl = cl - 1
    #     return subtree
    ##
    def sub_genomes_total_check1(self, tree):
        # optimize it by length
        lt = len(tree)
        if len(self.sub_genomes_total) == 0:
            self.sub_genomes_total['id'] = 0
        if not self.sub_genomes_total.__contains__(lt):
            value = self.sub_genomes_total['id'] + 1
            tree_dict = dict()
            tree_dict[tree] = value
            self.sub_genomes_total[lt] = tree_dict
        else:
            tree_dict = self.sub_genomes_total.get(lt)
            if not tree_dict.__contains__(tree):
                value = self.sub_genomes_total['id'] + 1
                tree_dict[tree] = value
                self.sub_genomes_total[lt] = tree_dict
            else:
                value = tree_dict.get(tree)
        return value

    def sub_genomes_total_check(self, tree):
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value

    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree = str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree

    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward(self):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            self.pop[gi] = ind

    def genome_to_mat(self, genome):
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            mat[:, cl] = column
        return mat

    def mat_to_pathes(self, mat):
        for cl in range(0, self.dbn.horizon, 1):
            self.tree_template[:, cl * 2] = mat[:, cl]
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        return pathes

    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes

    # ga_main
    def ga_main(self, public_idid):
        if self.progressbar:
            # bar = tqdm(total=int(self.parameters.values.get('generation_size')))
            pass
        f_list = [copy.deepcopy(self.pop)[i]['fitness'] for i in range(len(self.pop))]
        self.fits['max'].append(max(f_list))
        self.fits['min'].append(min(f_list))
        self.fits['mean'].append(sum(f_list) / len(f_list))
        self.fits['std'].append(np.std(np.array(f_list)))
        N = self.parameters.values.get('generation_size')
        bar = tqdm(total=N, position=0, desc='GA')
        for g in range(N):
            if not self.progressbar:
                # print("-- Generation %i --" % g)
                pass
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            nextoff = []
            while len(nextoff) != self.parameters.values.get('pop_size'):
                # Apply crossover and mutation on the offspring
                # Select two individuals
                offspring = [random.choice(selectpop) for i in range(0, 2)]
                if random.random() < self.parameters.values.get(
                        'crossover_rate'):  # cross two individuals with probability CXPB
                    crossoff = self.crossoperate(offspring)
                    if random.random() < self.parameters.values.get(
                            'mutation_rate'):  # mutate an individual with probability MUTPB
                        muteoff = self.mutation(crossoff)
                        if self.check_identity_flag:
                            if not self.check_identity(nextoff, muteoff) and not self.check_identity(self.pop,
                                                                                                     muteoff) and not self.check_identity(
                                self.pop_init, muteoff):
                                ind = self.gen_id(nextoff, self.gen_ind(muteoff))
                                nextoff.append(ind)  # initialize
                        else:
                            ind = self.gen_id(nextoff, self.gen_ind(muteoff))
                            nextoff.append(ind)  # initialize
            # The population is entirely replaced by the offspring
            # self.next_pop(nextoff)
            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, final_pop=True)
            else:
                self.next_pop(nextoff)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
            bar.update(1)
        bar.close()

        self.choose_group()
        # 将fitness结果存储
        self.result['Fitness'] = self.fits
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()

    # ==================================================================================================
    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd

    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)
        # print(len(individuals))
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen

    def selection_group(self, individuals, gp=None):
        return individuals

    def crossoperate(self, offspring):
        dim = len(offspring[0]['Gene'].data)
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
        # pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
        # pos2 = random.randrange(1, dim)
        pos1 = self.rand_pointer()  # select a position in the range from 0 to dim-1,
        pos2 = self.rand_pointer()
        newoff = Gene(data=[])  # offspring produced by cross operation
        temp = []
        if self.parameters.values.get('oddcross_mode'):
            for i in range(dim):
                if i % 2 == 1:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo1[i])
                    else:
                        temp.append(geninfo2[i])
                if i % 2 == 0:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo2[i])
                    else:
                        temp.append(geninfo1[i])
        else:
            for i in range(dim):
                if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                    temp.append(geninfo2[i])
                    # the gene data of offspring produced by cross operation is from the second offspring in the range [min(pos1,pos2),max(pos1,pos2)]
                else:
                    temp.append(geninfo1[i])
                    # the gene data of offspring produced by cross operation is from the frist offspring in the range [min(pos1,pos2),max(pos1,pos2)]
        newoff.data = temp
        return newoff

    def rand_pointer(self):
        if not self.parameters.values.get('weight_mode'):
            pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        else:
            sum_w = np.sum(self.weight)  #
            sum_ = 0
            u = random.random() * sum_w
            for pos in range(0, self.geneinfo_dim):
                sum_ += self.weight[pos]
                if sum_ > u:
                    break
        return pos

    def mutation(self, crossoff):
        pos = self.rand_pointer()  # chose a position in crossoff to perform mutation.
        crossoff.data[pos] = random.randint(0, self.dbn.num_as - 1)
        return crossoff

    def check_identity(self, pop, individual):
        # if the individual is already in the pop, then we don't need to add a copy of it
        gens = [ind['Gene'].data for ind in pop]
        # print(gens)
        for gen in gens:
            sum = 0
            for gi in range(0, len(gen)):
                sum = sum + np.abs(gen[gi] - individual.data[gi])
            if sum == 0:
                return True
        return False

    def gen_ind(self, muteoff):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        return ind

    def gen_id(self, nextoff, ind, gp=None):
        return ind

    def next_pop(self, nextoff, final_pop=None):
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            self.evaluate()
            [self.pop.append(ind) for ind in pop_temp]
            if final_pop:
                self.select_nextpop(self.parameters.values.get('tournament_size'))
            else:
                self.select_nextpop(self.parameters.values.get('pop_size'))
            # self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count == size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]

    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def choose_group(self):
        pass

    def gen_other(self):  # at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod = len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    if key == 'Fitness':
                        continue
                    for ki in self.result.get(key).keys():
                        if ki >= num_mod:
                            self.result.get(key).pop(ki)
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)

    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                rw, w, p = self.dbn.get_reward(modi=gi)
            else:
                rw, w, p = self.dbn.get_reward()
            # rw, w,p = self.dbn.get_reward()
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights, priors, rewards, policy_dicts

    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of ' + self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf')

    def display_pop(self, pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: ' + key)
                print('>value: ')
                print(ind.get(key))
            print("-----------------")


class PGA(object):
    def __init__(self, solver, filepath, result, dstr, num_mod):
        self.domain = dstr
        self.num_mod = num_mod
        self.solver = solver
        self.type = 'PGA'
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.pnames = Name()
        self.policy_dict_list = list()
        self.policy_path_weight_list = list()
        self.prior_belief_mat_list = list()
        self.group_fits_list = list()
        self.fits = list()
        self.ga_alg = []
        self.filepath = filepath

    # @ overide
    def ga_main(self):
        for g in range(0, self.parameters.values.get('group_size')):
            print("-- Group %i --" % g)
            # TF_set = self.gen_tf(g)
            self.ga_alg = GA(self.solver, self.filepath, self.result, self.domain, self.num_mod, self.type)
            self.ga_alg.ga_main()
            self.policy_dict_list.append(self.ga_alg.policy_dict)
            self.policy_path_weight_list.append(self.ga_alg.policy_path_weight)
            self.prior_belief_mat_list.append(self.ga_alg.prior_belief_mat)
            self.group_fits_list.append(self.ga_alg.fits)
        rerults = list()
        for r in self.group_fits_list:
            rerult = r.get('mean')
            rerults.append(rerult[len(rerults) - 1])
        index = np.argmax(np.array(rerults))
        self.result['policy_dict'] = self.policy_dict_list[index]
        self.result['policy_path_weight'] = self.policy_path_weight_list[index]
        self.result['policy_belief'] = self.prior_belief_mat_list[index]
        self.fits = self.group_fits_list[index]


class GGA(GA):
    def __init__(self, solver, filepath, result, dstr, num_mod):
        self.domain = dstr
        self.type = 'GGA'
        self.num_mod = num_mod
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.pnames = Name()
        super(GGA, self).__init__(solver, filepath, result, self.domain, num_mod, self.type)

    # @ overide
    def gen_pop(self):
        pop = []
        for gi in range(0, self.parameters.values.get('group_size')):
            for key in self.genomes.keys():
                geneinfo = self.genomes.get(key)
                fits = self.result['reward'].get(key)
                ind = self.gen_ind(Gene(data=geneinfo))
                ind['id'] = gi
                pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.evaluate()
        self.evaluate_group_criterion()
        self.group_data4plot()

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        for gp in range(0, self.parameters.values.get('group_size')):
            count = 0
            for ind in s_inds:
                if ind['id'] == gp:
                    nextpop.append(ind)  # store the chromosome and its fitness
                    count = count + 1
                if count == size:
                    break
        self.pop = nextpop

    def ga_main(self):
        print("Starting evolution")
        # Begin the evolution
        if self.progressbar:
            bar = tqdm(total=int(self.parameters.values.get('generation_size')))
        for g in range(self.parameters.values.get('generation_size')):
            if not self.progressbar:
                print("-- Generation %i --" % g)
            # Apply selection based on their converted fitness
            next_pops = []
            for gp in range(0, self.parameters.values.get('group_size')):
                # print("-- Group %i --" % gp)
                selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'), gp)
                nextoff = []
                # self.display_pop(selectpop)
                while len(nextoff) != self.parameters.values.get('pop_size'):
                    # Apply crossover and mutation on the offspring
                    # Select two individuals
                    offspring = [random.choice(selectpop) for i in range(0, 2)]
                    if random.random() < self.parameters.values.get(
                            'crossover_rate'):  # cross two individuals with probability CXPB
                        crossoff = self.crossoperate(offspring)
                        if random.random() < self.parameters.values.get(
                                'mutation_rate'):  # mutate an individual with probability MUTPB
                            muteoff = self.mutation(crossoff)
                            if self.check_identity_flag:
                                if not self.check_identity(nextoff, muteoff) and not self.check_identity(self.pop,
                                                                                                         muteoff):
                                    ind = self.gen_id(nextoff, self.gen_ind(muteoff), gp)
                                    nextoff.append(ind)  # initialize
                            else:
                                ind = self.gen_id(nextoff, self.gen_ind(muteoff), gp)
                                nextoff.append(ind)  # initialize
                [next_pops.append(ind) for ind in nextoff]
            # The population is entirely replaced by the offspring
            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(next_pops, final_pop=True)
            else:
                self.next_pop(next_pops)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            if not self.progressbar:
                print(
                    "Best individual found is %s, %s" % (
                        self.bestindividual['Gene'].data, self.bestindividual['fitness']))
                print("  Min fitness of current pop: %s" % min(fits))
                print("  Max fitness of current pop: %s" % max(fits))
                print("  Avg fitness of current pop: %s" % mean)
                print("  Std of currrent pop: %s" % std)
            else:
                bar.update(1)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
        if self.progressbar:
            bar.close()
        print("-- End of (successful) evolution --")

        self.choose_group()
        gens = [ind['Gene'].data for ind in self.pop]
        for gen in gens:
            print(gen)
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
            self.plot_groupdatas(self.group_fits, 'group_fits')
            self.plot_groupdatas(self.group_criterions, 'group_criterions')
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()
            self.plot_groupdatas_data(self.group_fits, 'group_fits')
            self.plot_groupdatas_data(self.group_criterions, 'group_criterions')

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        gpop = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            id = ind['id']
            if gpop.__contains__(id):
                ids = gpop.get(id)
                ids.append(gi)
                gpop[id] = ids
            else:
                gpop[id] = [gi]
        self.group_criterion = list()
        for key in gpop.keys():
            popindex = gpop.get(key)
            diversity_pop = self.cal_diversity(popindex)
            self.group_criterion.append(diversity_pop)
            for gi in popindex:
                ind = self.pop[gi]
                diversity_pop_gi = self.cal_diversity(popindex, gi)
                fits = diversity_pop_gi / diversity_pop  # divide
                ind['fitness'] = fits
                self.pop[gi] = ind

    def evaluate_distance(self):
        # evaluate the distance of pop
        # print('evaluate the distance of pop')
        genomes = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            genome = self.genome_template
            genome = [0 for i in range(0, len(genome))]
            count = 0
            for ind in self.pop:
                if ind['id'] == gp:
                    count = count + 1
                    gen = ind['Gene'].data
                    genome = [genome[i] + gen[i] for i in range(0, len(gen))]
            genome = [g / count for g in genome]
            genomes.append(genome)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            genome = genomes[ind['id']]
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    # pass implement
    def gen_id(self, nextoff, ind, gp):
        ind['id'] = gp
        return ind

    # def gen_id(self, nextoff, ind, gp):
    #     if random.random() >= self.parameters.values.get('emigrate_rate'):
    #         ind['id'] = gp
    #         return ind
    #     id_set = list(range(0, self.parameters.values.get('group_size')))
    #     id_set.pop(id_set.index(gp))
    #     ind['id'] = id_set[random.randrange(0, len(id_set))]
    #     return ind
    def gen_meanfits_deltas(self, nextoff):
        self.mean_fits = dict()
        self.deltas = dict()
        if nextoff is None:
            pop = self.pop
        else:
            pop = nextoff
        rate = 1 - self.parameters.values.get('emigrate_rate')
        if rate == 0:
            rate = 1 - self.parameters.values.get('emigrate_rate') + self.pnames.Elimit
        for gp in range(0, self.parameters.values.get('group_size')):
            fits = []
            for ind in pop:
                id = ind['id']
                if id == gp:
                    fits.append(ind['fitness'])
            fits = sorted(fits)
            # midind = int(np.round(len(fits)/2))
            # self.mean_fits.append(fits[midind])
            if len(fits) > 0:
                self.mean_fits[gp] = np.mean(fits)
                # a = np.abs(np.max(fits) - np.min(fits))
                # a =1
                # b = np.e
                # b = math.exp(1/rate)
                # delta = a*b/(np.abs(np.max(fits) - np.mean(fits))*rate)
                if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(3):
                    delta = np.log((1 - rate) / (rate)) / np.abs(np.max(fits) - np.mean(fits))
                if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(4):
                    delta = np.log((1 - 0.95 * rate) / (0.95 * rate)) / np.abs(np.max(fits) - np.mean(fits))
                # delta = a * b / (np.abs(np.max(fits) - np.mean(fits)) )
                self.deltas[gp] = delta
            else:
                self.mean_fits[gp] = 0
                self.deltas[gp] = 1
                print('123456789')
                for gpi in range(0, self.parameters.values.get('group_size')):
                    for ind in pop:
                        id = ind['id']
                        if id == gpi:
                            print(ind)

    def group_data4plot(self):
        for gp in range(0, self.parameters.values.get('group_size')):
            fits = []
            for ind in self.pop:
                id = ind['id']
                if id == gp:
                    fits.append(ind['fitness'])
            self.group_fits[gp].append(np.mean(fits))
            self.group_criterions[gp].append(self.group_criterion[gp])

    def plot_groupdatas_data(self, data, ylabel):
        pd = dict()
        xlen = len(self.group_fits[0])
        pd['xValues'] = range(0, xlen)
        minvalue = np.min([np.min(data.get(key)) for key in data.keys()])
        maxvalue = np.max([np.max(data.get(key)) for key in data.keys()])
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        pd['yValues'] = data
        pd['ykeys'] = data.keys()
        t_legend = list()
        for key in data.keys():
            t_legend.append('Group @' + str(key))
        pd['legend'] = t_legend
        pd['ylabel'] = ylabel
        pd['xlabel'] = 'generation'
        pd['title'] = 'The Fitness Converge Line of ' + self.type + '@ ' + ylabel
        pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The FCL of ' + self.type + '@ ' + ylabel + '-' + str(
            self.parameters.values.get("emigrate_method")) + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer') + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')][ylabel] = pd

    def plot_groupdatas(self, data, ylabel):
        import matplotlib.colors as mcolors
        fig = plt.figure()
        axis = fig.gca()
        xlen = len(self.group_fits[0])
        # xValues = range(0, self.parameters.values.get('generation_size'))
        xValues = range(0, xlen)
        minvalue = np.min([np.min(data.get(key)) for key in data.keys()])
        maxvalue = np.max([np.max(data.get(key)) for key in data.keys()])
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t = list()
        t_legend = list()
        count = 0
        color_set = mcolors.CSS4_COLORS
        ckeys = list(color_set.keys())
        for key in data.keys():
            t_i, = plt.plot(xValues, data.get(key), color=color_set[ckeys[0 + 3 * count + 10]])
            t.append(t_i)
            t_legend.append('Group @' + str(key))
            count = count + 1
        axis.set_ylabel(ylabel)
        axis.set_xlabel('generation')
        axis.grid()
        plt.legend(t, t_legend, loc='best')
        plt.title('The Fitness Converge Line of ' + self.type + '@ ' + ylabel)
        # plt.show()

        fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The FCL of ' + self.type + '@ ' + ylabel + '-' + str(
            self.parameters.values.get("emigrate_method")) + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer') + '.pdf')

    def next_pop(self, nextoff, final_pop=None):
        # print('next pop')
        # print(len(self.pop),len( nextoff))
        # self.emigration(self.pop)
        pop_temp = [ind for ind in self.pop]
        self.pop = [ind for ind in nextoff]
        self.evaluate()
        self.evaluate_group_criterion()
        nextoff = self.emigration(self.pop)
        self.pop = [ind for ind in pop_temp]
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            [self.pop.append(ind) for ind in pop_temp]
        else:
            self.pop = [ind for ind in nextoff]
        self.select_nextpop(self.parameters.values.get('pop_size'))
        self.evaluate_group_criterion()
        self.group_data4plot()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

    def emigration(self, nextoff):
        md = 1
        # random
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            # newoffs = list()
            inds = 0
            for ind in nextoff:
                if random.random() <= self.parameters.values.get('emigrate_rate'):
                    # newoff = self.gen_ind(ind['Gene'])
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds + 1
                    # newoff['id'] = gpn
                    # newoff['fitness'] = ind['fitness']
                    # newoffs.append(newoff)

            # [nextoff.append(ind) for ind in newoffs]
            # print(inds)
            return nextoff
        md = md + 1
        # random-bound
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            inds = 0
            gprobs = dict()
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = fit
                if gprobs.__contains__(gp):
                    probs = gprobs.get(gp)
                    probs[0].append(ind)
                    probs[1].append(prob)
                    gprobs[gp] = probs
                else:
                    gprobs[gp] = [[ind], [prob]]
            for g in gprobs.keys():
                number_eg = np.ceil(
                    self.parameters.values.get('emigrate_rate') * self.parameters.values.get('pop_size'))
                ind_list = gprobs.get(gp)[0]
                prob_list = gprobs.get(gp)[1]
                while number_eg > 0:
                    pos = self.russian_roulette(prob_list)
                    # print(pos, prob_list[pos])
                    ind = ind_list[pos]
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds + 1
                    number_eg = number_eg - 1
            # print(inds)
            return nextoff
        md = md + 1
        # sigmoid-bound
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            self.gen_meanfits_deltas(nextoff)
            newoffs = list()
            inds = 0
            gprobs = dict()
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if gprobs.__contains__(gp):
                    probs = gprobs.get(gp)
                    probs[0].append(ind)
                    probs[1].append(prob)
                    gprobs[gp] = probs
                else:
                    gprobs[gp] = [[ind], [prob]]
            for g in gprobs.keys():
                number_eg = np.ceil(
                    self.parameters.values.get('emigrate_rate') * self.parameters.values.get('pop_size'))
                ind_list = gprobs.get(gp)[0]
                prob_list = gprobs.get(gp)[1]
                while number_eg > 0:
                    pos = self.russian_roulette(prob_list)
                    # print(pos, prob_list[pos])
                    ind = ind_list[pos]
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds + 1
                    number_eg = number_eg - 1
            # print(inds)
            if len(newoffs) > 0:
                # print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff
        md = md + 1
        # sigmoid- cutoff
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            self.gen_meanfits_deltas(nextoff)
            # pop_temp = [ind for ind in self.pop]
            # self.pop = [ind for ind in nextoff]
            # self.evaluate()
            # self.evaluate_group_criterion()
            # self.pop = [ind for ind in  pop_temp]
            newoffs = list()
            # counts = dict()
            for ind in nextoff:
                gp = ind['id']
            #     if counts.__contains__(gp):
            #         counts[gp] = counts.get(gp)+1
            #     else:
            #         counts[gp] = 1
            # if self.parameters.values.get('pelite_mode'):
            #     limit = self.parameters.values.get('pop_size')
            #     if self.parameters.values.get('elite_mode'):
            #         limit = self.parameters.values.get('tournament_size')
            # else:
            #     limit = int(np.round(self.parameters.values.get('pop_size'))*0.618)
            inds = 0
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                # if counts.get(gp) <= limit:
                #     continue
                prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if prob >= 1 - self.parameters.values.get('emigrate_rate'):
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != gp:
                        # newoff = self.gen_ind(ind['Gene'])
                        # newoff['id'] = gpn
                        # newoff['fitness'] = ind['fitness']
                        # newoffs.append(newoff)
                        ind['id'] = gpn
                        inds = inds + 1
                        # counts[gp] = counts.get(gp) -1
                        # counts[gpn] = counts.get(gpn) + 1
                        # print('>>>emigrate from: ' + str(gp) + '  to: ' + str(gpn))
            # print(inds)
            if len(newoffs) > 0:
                # print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff
        md = md + 1
        # sigmoid- group-cutoff
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            fits = [ind['fitness'] for ind in nextoff]
            mean_fits = np.mean(fits)
            # midind = int(np.round(len(fits) / 2))
            # mean_fits =  fits[midind]
            rate = 1 - self.parameters.values.get('emigrate_rate')
            if rate == 0:
                rate = 1 - self.parameters.values.get('emigrate_rate') + self.pnames.Elimit
            # delta = np.abs(np.max(fits) -np.min(fits)) *np.e / (np.abs(np.max(fits) - mean_fits) * rate)
            delta = np.log((1 - rate) / (rate)) / np.abs(np.max(fits) - np.mean(fits))
            newoffs = list()
            inds = 0
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = self.stable_sigmoid((fit - mean_fits) * delta)
                # prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if prob >= 1 - self.parameters.values.get('emigrate_rate'):
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != gp:
                        # newoff = self.gen_ind(ind['Gene'])
                        # newoff['id'] = gpn
                        # newoff['fitness'] = ind['fitness']
                        # newoffs.append(newoff)
                        ind['id'] = gpn
                        inds = inds + 1
                        # print('>>>emigrate from: ' + str(gp) + '  to: ' + str(gpn))
            # print(inds)
            if len(newoffs) > 0:
                # print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff

    def russian_roulette(self, gc):
        import random
        min_gc = np.min(gc)
        gctt = [e - min_gc + self.pnames.Elimit for e in gc]
        sum_w = np.sum(gctt)
        sum_ = 0
        u = random.random() * sum_w
        for pos in range(0, len(gctt)):
            sum_ += gctt[pos]
            if sum_ > u:
                break
        return pos

    # Sigmoid Function
    def stable_sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig

    def choose_group(self):
        self.next_pop(self.pop)
        self.evaluate_group_criterion()
        g_choosen = np.argmax(np.array(self.group_criterion)) + 1
        self.pop_save = [ind for ind in self.pop]
        pop = [ind for ind in self.pop if g_choosen == ind['id']]
        self.pop = pop

    def selection_group(self, individuals, gp):
        individuals = [ind for ind in individuals if ind['id'] == gp]
        return individuals

    def initialize_other(self):
        self.pop_save = []
        self.group_criterion = list()
        self.group_fits = dict()
        self.group_criterions = dict()
        for gp in range(0, self.parameters.values.get('group_size')):
            self.group_fits[gp] = list()
            self.group_criterions[gp] = list()

    # new
    def evaluate_group_criterion(self):
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(1):
            self.evaluate_gc_distance()
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(2):
            self.evaluate_gc_diversity()
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(3):
            self.evaluate_gc_reward()

    def evaluate_gc_distance(self):
        genomes = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            genome = self.genome_template
            genome = [0 for i in range(0, len(genome))]
            count = 0
            for ind in self.pop:
                if ind['id'] == gp:
                    count = count + 1
                    gen = ind['Gene'].data
                    genome = [genome[i] + gen[i] for i in range(0, len(gen))]
            genome = [g / count for g in genome]
            genomes.append(genome)
        distances = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            genome = genomes[ind['id']]
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            distances.append(fits)
        self.group_criterion = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            reward_pop = []
            for gi in range(0, len(self.pop)):
                ind = self.pop[gi]
                id = ind['id']
                if id == gp:
                    reward_pop.append(distances[gi])
            self.group_criterion.append(np.mean(np.array(reward_pop)))

    def evaluate_gc_reward(self):
        reward = list()
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The  Policy Tree of ' + self.type, self.dbn.action_list,
                                     self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            # if self.parameters.values.get('cover_mode'):
            #     ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            # else:
            #     ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            r, w, p = self.dbn.get_reward()
            reward.append(r)
        self.group_criterion = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            reward_pop = []
            for gi in range(0, len(self.pop)):
                ind = self.pop[gi]
                id = ind['id']
                if id == gp:
                    reward_pop.append(reward[gi])
            self.group_criterion.append(np.mean(np.array(reward_pop)))

    def evaluate_gc_diversity(self):
        # evaluate the distance or diversity of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        gpop = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            id = ind['id']
            if gpop.__contains__(id):
                ids = gpop.get(id)
                ids.append(gi)
                gpop[id] = ids
            else:
                gpop[id] = [gi]
        self.group_criterion = list()
        for key in gpop.keys():
            popindex = gpop.get(key)
            diversity_pop = self.cal_diversity(popindex)
            self.group_criterion.append(diversity_pop)


class PM(object):
    def __init__(self, solver, filepath, result, dstr, num_mod):
        self.domain = dstr
        self.type = 'PM'
        # print('Initializing ' + self.type)
        self.num_mod = num_mod
        self.pnames = Name()
        self.solver = solver
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)

        self.progressbar = True
        self.position = dict()
        self.velocity = dict()
        self.initialize()
        self.initialize_pop()
        self.alg_name = ''
        self.plot_flag = False

    def set_alg_name(self, name):
        self.alg_name = name

    def initialize(self):
        self.genomes = dict()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()

    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path

    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path

    def fill_tree_template(self):
        # self.tree_template = np.zeros([self.get_num_path(), self.get_len_path()])
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                # pathes = self.result['policy_dict'].get(key)
                self.tree_template = self.result['policy_dict'].get(key)
                break
            # for i in range(0, self.get_num_path()):
            #     self.tree_template[i, :] = pathes[i]
        # print(self.tree_template)

    def initialize_other(self):
        pass

    def initialize_pop(self):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate()  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'velocity': 0}

    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()
        self.gen_weight()

    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)

    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for j in range(0, self.get_len_path(), 2):
            hi = int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def pathes_to_mat(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        for i in range(0, self.get_num_path()):
            mat[i, :] = [pathes[i][j] for j in range(0, self.get_len_path(), 2)]
        return mat

    def mat_to_genome(self, mat):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.get_num_observation(), self.dbn.horizon - cl - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(mat[rw, cl]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def gen_genome_level(self):
        '''
             a	o	a	o	a        a	a  a
             2	0	0	0	2        2  0  2
             2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0 =>[3 2 2 1 1 1 1]
             2	1	2	0	2        2  2  2
             2	1	2	1	0        2  2  0
             '''
        if self.genome_template == None:
            print('============errrrrr')
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.dbn.horizon - cl
            start = start + num
        self.genome_level = genome_level
        # print('-------------------------------')
        # print(genome_level)

    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.dbn.horizon - 1):
            step = np.power(self.dbn.num_os, cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.dbn.num_os)
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
        # print('-------------------------------')
        # print(self.genome_arc)

    def gen_weight(self):
        if not self.parameters.values.get('weight_mode'):
            pass
        w = 1 / self.dbn.horizon
        level = self.genome_level
        self.weight = np.array([w / (level.count(level[i])) for i in range(0, len(level))])
        # print(self.weight)

    def gen_pop(self):
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo), v=0)
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]

    def evaluate(self):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()

    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def sub_genomes_total_check1(self, tree):
        # optimize it by length
        lt = len(tree)
        if len(self.sub_genomes_total) == 0:
            self.sub_genomes_total['id'] = 0
        if not self.sub_genomes_total.__contains__(lt):
            value = self.sub_genomes_total['id'] + 1
            tree_dict = dict()
            tree_dict[tree] = value
            self.sub_genomes_total[lt] = tree_dict
        else:
            tree_dict = self.sub_genomes_total.get(lt)
            if not tree_dict.__contains__(tree):
                value = self.sub_genomes_total['id'] + 1
                tree_dict[tree] = value
                self.sub_genomes_total[lt] = tree_dict
            else:
                value = tree_dict.get(tree)
        return value

    def sub_genomes_total_check(self, tree):
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value

    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree = str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree

    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward(self):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            self.pop[gi] = ind

    def genome_to_mat(self, genome):
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            mat[:, cl] = column
        return mat

    def mat_to_pathes(self, mat):
        for cl in range(0, self.dbn.horizon, 1):
            self.tree_template[:, cl * 2] = mat[:, cl]
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        return pathes

    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes

    # ga_main
    def ga_main(self, public_idid):
        # initial v
        self.init_v(len(self.pop), self.geneinfo_dim)
        self.init_p(self.pop)

        # 标记出本次迭代最好粒子
        self.bestindividual = Common.get_pbest(copy.deepcopy(self.pop))
        self.pbest = Common.get_pbest(copy.deepcopy(self.pop))

        f_list = [copy.deepcopy(self.pop)[i]['fitness'] for i in range(len(self.pop))]
        self.fits['max'].append(max(f_list))
        self.fits['min'].append(min(f_list))
        self.fits['mean'].append(sum(f_list) / len(f_list))
        self.fits['std'].append(np.std(np.array(f_list)))

        N = self.parameters.values.get('generation_size')
        bar = tqdm(total=N, position=0, desc='PM')

        for g in range(N):
            self.update_weight(g)
            if not self.progressbar:
                print("-- Generation %i --" % g)
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            nextoff = self.get_pso_pop(selectpop)

            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, final_pop=True)
            else:
                self.next_pop(nextoff)

            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            self.pbest = self.selectBest(self.pop)
            if self.pbest['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = copy.deepcopy(self.pbest)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
            bar.update(1)
        bar.close()
        self.result['Fitness'] = self.fits
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()

    def init_v(self, num, dim):
        range_dim = list(np.arange(dim))
        action_dim = list(np.arange(self.dbn.num_as))
        for i in range(num):
            v_list = []
            p = random.sample(range_dim, 1)[0]
            v_list.append(p)
            a = random.sample(action_dim, 1)[0]
            v_list.append(a)
            r = random.randint(1, 10) / 10
            v_list.append(r)
            self.velocity[i] = [v_list]
            self.pop[i]['velocity'] = [v_list]

    def init_p(self, selectpop):
        pop_num = len(selectpop)
        for i in range(pop_num):
            self.position[i] = selectpop[i]['Gene'].data

    def update_weight(self, g):
        LDW_ini = 0.9
        LDW_end = 0.4
        iter = self.parameters.values['generation_size']
        self.weight_pso = ((LDW_ini - LDW_end) * (iter - g) / iter) + LDW_end

    def get_pso_pop(self, selectpop):
        nextoff = []
        pbest_data = self.pbest['Gene'].data
        gbest_data = self.bestindividual['Gene'].data
        for key in range(len(selectpop)):
            p_v_dict = dict()
            # 获取每个粒子的位置
            data = selectpop[key]['Gene'].data.copy()
            w, c1, c2 = self.form_par(self.weight_pso, self.parameters.values['learning_rate1'],
                                      self.parameters.values['learning_rate1'])
            # 通过换位减获取速度
            pop_v0 = self.update_v0(w, selectpop[key]['velocity'])
            pop_v1 = self.update_v_by_p(c1, pbest_data, data)
            pop_v2 = self.update_v_by_p(c2, gbest_data, data)

            update_v = self.pop_v_mul(pop_v0, pop_v1, pop_v2)
            # 更新速度
            # self.velocity[key] = update_v
            # selectpop[key]['velocity'] = update_v
            p_v_dict['velocity'] = update_v
            # 更新粒子位置
            pos_ = self.update_pos(data, update_v)
            # self.position[key] = pos_
            p_v_dict['position'] = pos_
            # selectpop[key]['Gene'].data = pos_
            nextoff.append(p_v_dict)
        off = self.gen_Gene(nextoff)
        return off

    def form_par(self, w, c1, c2):
        # array = np.array([w, c1, c2])
        # r1 = random.random()
        # r2 = random.random()
        array = np.array([w, c1, c2])
        sum = np.sum(array)
        for i in range(len(array)):
            array[i] = array[i] / sum
        return array[0], array[1], array[2]

    def pop_v_mul(self, v1, v2, v3):
        v1_c = v1.copy()
        v2_c = v2.copy()
        v3_c = v3.copy()
        v_add = self.add_v(v1_c, v2_c)
        v = self.add_v(v_add, v3_c)
        return v

    def add_v(self, v1_c, v2_c):
        for i in range(len(v1_c)):
            for j in range(len(v2_c)):
                if v1_c[i][0] == v2_c[j][0]:
                    if v1_c[i][1] == v2_c[j][1]:
                        v2_c[j][2] = v1_c[i][2] + v2_c[j][2]
                        v1_c[i][2] = 100
        v_ = []
        for i in range(len(v1_c)):
            if v1_c[i][2] != 100:
                v_.append(v1_c[i])
        for i in range(len(v2_c)):
            v_.append(v2_c[i])
        for i in v_:
            i[2] = round(i[2], 2)
        return v_

    def update_v0(self, w, v):
        v = copy.deepcopy(v)
        for i in range(len(v)):
            v[i][2] = v[i][2] * w
        return v

    def gen_Gene(self, nextoff):
        popoff = []
        for i in range(len(nextoff)):
            newoff = Gene(data=[])
            newoff.data = nextoff[i]['position']
            v = nextoff[i]['velocity']
            ind = self.gen_id(popoff, self.gen_ind(newoff, v))
            popoff.append(ind)
        return popoff

    def update_pos(self, data, pop_v):
        for key in pop_v:
            if random.random() <= key[2] or key[2] > 1:
                data[key[0]] = key[1]
            else:
                data[key[0]] = random.sample(list(np.arange(self.dbn.num_as)), 1)[0]
        return data

    def update_v_by_p(self, c, best_data, data):
        c = c * random.random()
        list_all = []
        j = 0
        for i in range(len(data)):
            if data[i] == best_data[i]:
                j += 1
                continue
            else:
                v_t = [i, best_data[i], c]
                list_all.append(v_t)
        if j == len(data):
            range_dim = list(np.arange(self.geneinfo_dim))
            action_dim = list(np.arange(self.dbn.num_as))
            list_all.append(
                [random.sample(range_dim, 1)[0], random.sample(action_dim, 1)[0], random.randint(1, 10) / 10])
        return list_all

    # ==================================================================================================
    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd

    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)
        # print(len(individuals))
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen

    def selection_group(self, individuals, gp=None):
        return individuals

    def rand_pointer(self):
        if not self.parameters.values.get('weight_mode'):
            pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        else:
            sum_w = np.sum(self.weight)  #
            sum_ = 0
            u = random.random() * sum_w
            for pos in range(0, self.geneinfo_dim):
                sum_ += self.weight[pos]
                if sum_ > u:
                    break
        return pos

    def gen_ind(self, muteoff, v):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        ind['velocity'] = v
        return ind

    def gen_id(self, nextoff, ind, gp=None):
        return ind

    def next_pop(self, nextoff, final_pop=None):
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            self.evaluate()
            [self.pop.append(ind) for ind in pop_temp]
            if final_pop:
                self.select_nextpop(self.parameters.values.get('tournament_size'))
            else:
                self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

        # self.pop = [ind for ind in nextoff]
        # self.evaluate()

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count == size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]

    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def gen_other(self):  # at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod = len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    for ki in self.result.get(key).keys():
                        if key == 'Fitness':
                            continue
                        if ki >= num_mod:
                            self.result.get(key).pop(ki)
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)

    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                rw, w, p = self.dbn.get_reward(modi=gi)
            else:
                rw, w, p = self.dbn.get_reward()
            # rw, w,p = self.dbn.get_reward()
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights, priors, rewards, policy_dicts

    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of ' + self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf')

    def display_pop(self, pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: ' + key)
                print('>value: ')
                print(ind.get(key))
            print("-----------------")


class ACO(object):
    def __init__(self, solver, filepath, result, dstr, num_mod, type=None):
        if type is None:
            self.type = 'ACO'
        else:
            self.type = type
        # print('Initializing ' + self.type)
        self.plot_flag = False  # for GUI GEN EXE
        self.domain = dstr
        self.pnames = Name()
        self.solver = solver
        self.num_mod = num_mod
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.result = result
        self.check_identity_flag = True
        self.progressbar = True
        self.initialize()
        self.initialize_pop()
        self.alg_name = ''

    def set_alg_name(self, name):
        self.alg_name = name

    # initialize
    def initialize(self):
        self.genomes = dict()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()

    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path

    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path

    def fill_tree_template(self):
        # self.tree_template = np.zeros([self.get_num_path(), self.get_len_path()])
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                # pathes = self.result['policy_dict'].get(key)
                self.tree_template = self.result['policy_dict'].get(key)
                break
            # for i in range(0, self.get_num_path()):
            #     self.tree_template[i, :] = pathes[i]
        # print(self.tree_template)

    def initialize_other(self):
        pass

    def initialize_pop(self):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate()  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'id': 0}

    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()
        self.gen_weight()

    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)

    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for j in range(0, self.get_len_path(), 2):
            hi = int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def pathes_to_mat(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        for i in range(0, self.get_num_path()):
            mat[i, :] = [pathes[i][j] for j in range(0, self.get_len_path(), 2)]
        return mat

    def mat_to_genome(self, mat):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.get_num_observation(), self.dbn.horizon - cl - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(mat[rw, cl]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def gen_genome_level(self):
        '''
             a	o	a	o	a        a	a  a
             2	0	0	0	2        2  0  2
             2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0 =>[3 2 2 1 1 1 1]
             2	1	2	0	2        2  2  2
             2	1	2	1	0        2  2  0
             '''
        if self.genome_template == None:
            print('============errrrrr')
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.dbn.horizon - cl
            start = start + num
        self.genome_level = genome_level
        # print('-------------------------------')
        # print(genome_level)

    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.dbn.horizon - 1):
            step = np.power(self.dbn.num_os, cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.dbn.num_os)
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
        # print('-------------------------------')
        # print(self.genome_arc)

    def gen_weight(self):
        if not self.parameters.values.get('weight_mode'):
            pass
        w = 1 / self.dbn.horizon
        level = self.genome_level
        self.weight = np.array([w / (level.count(level[i])) for i in range(0, len(level))])
        # print(self.weight)

    def gen_pop(self):
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo))
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]

    def evaluate(self):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()

    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def sub_genomes_total_check1(self, tree):
        # optimize it by length
        lt = len(tree)
        if len(self.sub_genomes_total) == 0:
            self.sub_genomes_total['id'] = 0
        if not self.sub_genomes_total.__contains__(lt):
            value = self.sub_genomes_total['id'] + 1
            tree_dict = dict()
            tree_dict[tree] = value
            self.sub_genomes_total[lt] = tree_dict
        else:
            tree_dict = self.sub_genomes_total.get(lt)
            if not tree_dict.__contains__(tree):
                value = self.sub_genomes_total['id'] + 1
                tree_dict[tree] = value
                self.sub_genomes_total[lt] = tree_dict
            else:
                value = tree_dict.get(tree)
        return value

    def sub_genomes_total_check(self, tree):
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value

    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree = str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree

    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward(self):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            self.pop[gi] = ind

    def genome_to_mat(self, genome):
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            mat[:, cl] = column
        return mat

    def mat_to_pathes(self, mat):
        for cl in range(0, self.dbn.horizon, 1):
            self.tree_template[:, cl * 2] = mat[:, cl]
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        return pathes

    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes

    # ga_main
    def ga_main(self, public_idid):
        if self.progressbar:
            # bar = tqdm(total=int(self.parameters.values.get('generation_size')))
            pass
        f_list = [copy.deepcopy(self.pop)[i]['fitness'] for i in range(len(self.pop))]
        self.fits['max'].append(max(f_list))
        self.fits['min'].append(min(f_list))
        self.fits['mean'].append(sum(f_list) / len(f_list))
        self.fits['std'].append(np.std(np.array(f_list)))
        # 定义信息素浓度集合
        self.xinxisu_dict = dict()
        n = self.geneinfo_dim
        action_num = self.dbn.num_as
        for i in range(n):
            l = list()
            for j in range(action_num):
                l.append([j, 0])
            self.xinxisu_dict[i] = l

        N = self.parameters.values.get('generation_size')
        bar = tqdm(total=N, position=0, desc='ACO')

        for g in range(N):
            if not self.progressbar:
                # print("-- Generation %i --" % g)
                pass
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            # ACO具体实现
            nextoff = self.get_aco_pop(selectpop)

            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, final_pop=True)
            else:
                self.next_pop(nextoff)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind

            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
            bar.update(1)
        bar.close()
        self.choose_group()
        # 将fitness结果存储
        self.result['Fitness'] = self.fits
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()

    def get_aco_pop(self, selectpop):
        # 将selectpop中的data转换格式
        '''
        [2,2,2,1,0,1,0] ---->[[0,2],[1,2],[2,2],[3,1],[4,0],[5,1],[6,0]]
        '''
        selectpop_copy = copy.deepcopy(selectpop)
        selectpop_copy2 = self.change_pop(selectpop_copy)
        '''
        计算信息素
        '''
        pheropop = self.cal_phero(selectpop_copy2)
        return pheropop

    def cal_phero(self, selectpop):
        '''
        1.先将fitness值放进data中
        '''
        for i in range(len(selectpop)):
            data = selectpop[i]['Gene'].data
            fit = selectpop[i]['fitness']
            if len(data[0]) == 3:
                continue
            for j in range(len(data)):
                data[j].append(fit)
        '''
        2.计算信息素的和
        '''
        phero_list = list()
        for i in range(len(selectpop)):
            data = copy.deepcopy(selectpop[i])['Gene'].data
            for d in data:
                phero_list.append(d)
        data_all = self.quchong(phero_list)
        # 合并data_all进信息素字典
        self.merge_data(data_all)

        '''
        3.根据信息素去更新数据
        '''
        antdata_dict = self.update_data()
        '''
        4.将数据封装成Gene
        '''
        next_off = self.gen_Gene(antdata_dict)

        # '''
        # 3.获得每个路径的信息素
        # '''
        # # 存对象地址
        # address_list = list()
        # for i in range(len(selectpop)):
        #     if id(selectpop[i]) in address_list:
        #         continue
        #     else:
        #         address_list.append(id(selectpop[i]))
        #         # 更新信息素浓度为fitness+sum_fitness
        #         data = selectpop[i]['Gene'].data
        #         for j in range(len(data)):
        #             dj = data[j]
        #             for k in range(len(data_all)):
        #                 if dj[0] == data_all[k][0] and dj[1] == data_all[k][1]:
        #                     dj[2] = dj[2] + data_all[k][2]
        #                     break
        #         # 将每个蚂蚁的信息素浓度转化为概率
        return next_off

    def gen_Gene(self, ant_data):
        popoff = []
        for i in range(len(ant_data)):
            newoff = Gene(data=[])
            newoff.data = ant_data[i]
            ind = self.gen_id(popoff, self.gen_ind(newoff))
            popoff.append(ind)
        return popoff

    def update_data(self):
        ant_num = self.num_mod
        ant_dict = dict()
        trans_rate = self.parameters.values.get('trans_rate')
        for i in range(ant_num):
            a_list = list()
            for j in range(self.geneinfo_dim):
                p_r = random.random()
                if p_r < trans_rate:
                    # 随机选择一个动作
                    a = random.sample(list(np.arange(self.dbn.num_as)), 1)[0]
                    a_list.append(a)
                else:
                    # 选择信息素最大的action
                    x_max = self.max_xinxisu(self.xinxisu_dict[j])
                    a_list.append(x_max)
            ant_dict[i] = a_list
        return ant_dict

    def max_xinxisu(self, data):
        max_d = 0
        index = 0
        for d in data:
            if d[1] > max_d:
                max_d = d[1]
                index = d[0]
        return index

    def merge_data(self, data_all):
        rho_rate = self.parameters.values.get('rho_rate')
        for key in self.xinxisu_dict.keys():
            k_d = self.xinxisu_dict[key]
            for i in range(len(data_all)):
                if key == data_all[i][0]:
                    for j in k_d:
                        if j[0] == data_all[i][1]:
                            j[1] = rho_rate * j[1] + data_all[i][2]
                            break
                    # break

    def quchong(self, d):
        quchong_list_sum = list()
        for j in range(len(d)):
            d_j = d[j]
            if quchong_list_sum:
                flag = False
                for i in range(len(quchong_list_sum)):
                    data_i = quchong_list_sum[i]
                    if d_j[0] == data_i[0] and d_j[1] == data_i[1]:
                        quchong_list_sum[i][2] = quchong_list_sum[i][2] + d_j[2]
                        flag = True
                        break
                if flag == False:
                    quchong_list_sum.append(d_j)
            else:
                quchong_list_sum.append(d_j)
        return quchong_list_sum

    def change_pop(self, selectpop):
        for i in range(len(selectpop)):
            data = copy.deepcopy(selectpop[i]['Gene'].data)
            if np.array(data).ndim != 1:
                continue
            data_list = list()
            for j in range(len(data)):
                data_list.append([j, data[j]])
            selectpop[i]['Gene'].data = data_list
        return selectpop

    # ==================================================================================================
    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd

    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)
        # print(len(individuals))
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen

    def selection_group(self, individuals, gp=None):
        return individuals

    def crossoperate(self, offspring):
        dim = len(offspring[0]['Gene'].data)
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
        # pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
        # pos2 = random.randrange(1, dim)
        pos1 = self.rand_pointer()  # select a position in the range from 0 to dim-1,
        pos2 = self.rand_pointer()
        newoff = Gene(data=[])  # offspring produced by cross operation
        temp = []
        if self.parameters.values.get('oddcross_mode'):
            for i in range(dim):
                if i % 2 == 1:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo1[i])
                    else:
                        temp.append(geninfo2[i])
                if i % 2 == 0:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo2[i])
                    else:
                        temp.append(geninfo1[i])
        else:
            for i in range(dim):
                if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                    temp.append(geninfo2[i])
                    # the gene data of offspring produced by cross operation is from the second offspring in the range [min(pos1,pos2),max(pos1,pos2)]
                else:
                    temp.append(geninfo1[i])
                    # the gene data of offspring produced by cross operation is from the frist offspring in the range [min(pos1,pos2),max(pos1,pos2)]
        newoff.data = temp
        return newoff

    def rand_pointer(self):
        if not self.parameters.values.get('weight_mode'):
            pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        else:
            sum_w = np.sum(self.weight)  #
            sum_ = 0
            u = random.random() * sum_w
            for pos in range(0, self.geneinfo_dim):
                sum_ += self.weight[pos]
                if sum_ > u:
                    break
        return pos

    def mutation(self, crossoff):
        pos = self.rand_pointer()  # chose a position in crossoff to perform mutation.
        crossoff.data[pos] = random.randint(0, self.dbn.num_as - 1)
        return crossoff

    def check_identity(self, pop, individual):
        # if the individual is already in the pop, then we don't need to add a copy of it
        gens = [ind['Gene'].data for ind in pop]
        # print(gens)
        for gen in gens:
            sum = 0
            for gi in range(0, len(gen)):
                sum = sum + np.abs(gen[gi] - individual.data[gi])
            if sum == 0:
                return True
        return False

    def gen_ind(self, muteoff):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        return ind

    def gen_id(self, nextoff, ind, gp=None):
        return ind

    def next_pop(self, nextoff, final_pop=None):
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            self.evaluate()
            [self.pop.append(ind) for ind in pop_temp]
            if final_pop:
                self.select_nextpop(self.parameters.values.get('tournament_size'))
            else:
                self.select_nextpop(self.parameters.values.get('pop_size'))
            # self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count == size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]

    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def choose_group(self):
        pass

    def gen_other(self):  # at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod = len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    if key == 'Fitness':
                        continue
                    for ki in self.result.get(key).keys():
                        if ki >= num_mod:
                            self.result.get(key).pop(ki)
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)

    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                rw, w, p = self.dbn.get_reward(modi=gi)
            else:
                rw, w, p = self.dbn.get_reward()
            # rw, w,p = self.dbn.get_reward()
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights, priors, rewards, policy_dicts

    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of ' + self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf')

    def display_pop(self, pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: ' + key)
                print('>value: ')
                print(ind.get(key))
            print("-----------------")


class PSOC(object):
    def __init__(self, solver, filepath, result, dstr, num_mod, public_idid):
        self.domain = dstr
        self.type = 'PSOC'
        # print('Initializing ' + self.type)
        self.num_mod = num_mod
        self.pnames = Name()
        self.solver = solver
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)

        self.progressbar = True
        self.position = dict()
        self.velocity = dict()
        self.initialize()
        self.initialize_pop(public_idid)
        self.alg_name = ''
        self.plot_flag = False
        self.policyList = None
        self.policyPathWeight = None
        self.temp_pop = None

    def set_alg_name(self, name):
        self.alg_name = name

    def initialize(self):
        self.genomes = dict()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()

    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path

    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path

    def fill_tree_template(self):
        # self.tree_template = np.zeros([self.get_num_path(), self.get_len_path()])
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                # pathes = self.result['policy_dict'].get(key)
                self.tree_template = self.result['policy_dict'].get(key)
                break
            # for i in range(0, self.get_num_path()):
            #     self.tree_template[i, :] = pathes[i]
        # print(self.tree_template)

    def initialize_other(self):
        pass

    def initialize_pop(self, public_idid):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate(public_idid)  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'velocity': 0}

    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()
        self.gen_weight()

    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)

    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for j in range(0, self.get_len_path(), 2):
            hi = int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def pathes_to_mat(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        for i in range(0, self.get_num_path()):
            mat[i, :] = [pathes[i][j] for j in range(0, self.get_len_path(), 2)]
        return mat

    def mat_to_genome(self, mat):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.get_num_observation(), self.dbn.horizon - cl - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(mat[rw, cl]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def gen_genome_level(self):
        '''
             a	o	a	o	a        a	a  a
             2	0	0	0	2        2  0  2
             2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0 =>[3 2 2 1 1 1 1]
             2	1	2	0	2        2  2  2
             2	1	2	1	0        2  2  0
             '''
        if self.genome_template == None:
            print('============errrrrr')
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.dbn.horizon - cl
            start = start + num
        self.genome_level = genome_level
        # print('-------------------------------')
        # print(genome_level)

    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.dbn.horizon - 1):
            step = np.power(self.dbn.num_os, cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.dbn.num_os)
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
        # print('-------------------------------')
        # print(self.genome_arc)

    def gen_weight(self):
        if not self.parameters.values.get('weight_mode'):
            pass
        w = 1 / self.dbn.horizon
        level = self.genome_level
        self.weight = np.array([w / (level.count(level[i])) for i in range(0, len(level))])
        # print(self.weight)

    def gen_pop(self):
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo), v=0)
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]

    def evaluate(self, public_idid):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(4):
            self.evaluate_entropy(public_idid)
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(5):
            self.evaluate_diversity_entropy(public_idid)

    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def sub_genomes_total_check1(self, tree):
        # optimize it by length
        lt = len(tree)
        if len(self.sub_genomes_total) == 0:
            self.sub_genomes_total['id'] = 0
        if not self.sub_genomes_total.__contains__(lt):
            value = self.sub_genomes_total['id'] + 1
            tree_dict = dict()
            tree_dict[tree] = value
            self.sub_genomes_total[lt] = tree_dict
        else:
            tree_dict = self.sub_genomes_total.get(lt)
            if not tree_dict.__contains__(tree):
                value = self.sub_genomes_total['id'] + 1
                tree_dict[tree] = value
                self.sub_genomes_total[lt] = tree_dict
            else:
                value = tree_dict.get(tree)
        return value

    def sub_genomes_total_check(self, tree):
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value

    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree = str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree

    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward(self):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward()
            self.pop[gi] = ind

    def evaluate_entropy(self, public_idid):
        weights, policy_dicts = self.getPathAndWeight()
        # 获得信息熵
        idid = Common.get_idid(public_idid, policy_dicts, weights)
        path_pr_list = Common.cal_path_pr(idid)
        entropy = Common.cal_entropy(path_pr_list)
        # entropy = Common.cal_entropy(Common.get_idid(public_idid, policy_dicts, weights))
        # entropy = Common.cal_xinxiliang(public_idid, policy_dicts, weights)
        # 替换self.pop中的fitness
        for i in range(len(entropy)):
            self.pop[i]["fitness"] = entropy[i]

    def getPathAndWeight(self):
        weights = dict()
        policy_dicts = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list,
                                     self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            # if self.parameters.values.get('cover_mode'):
            #     rw, w, p = self.dbn.get_reward(modi=gi)
            # else:
            #     rw, w, p = self.dbn.get_reward()
            rw, w, p = self.dbn.get_reward()
            weights[gi] = w
            policy_dicts[gi] = pathes
        return weights, policy_dicts

    def getPathAndWeight2(self, pop):
        weights = dict()
        policy_dicts = dict()
        for gi in range(0, len(pop)):
            ind = pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list,
                                     self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            # if self.parameters.values.get('cover_mode'):
            #     rw, w, p = self.dbn.get_reward(modi=gi)
            # else:
            #     rw, w, p = self.dbn.get_reward()
            rw, w, p = self.dbn.get_reward()
            weights[gi] = w
            policy_dicts[gi] = pathes
        return weights, policy_dicts

    def evaluate_diversity_entropy(self, public_idid):
        # 1、计算diversity
        diversity_list = self.get_d()
        # 1.1、标准化多样性
        diversity_norm = Common.norm_diversity(diversity_list)
        # 2、计算entropy
        weights, policy_dicts = self.getPathAndWeight()
        idid = Common.get_idid(public_idid, policy_dicts, weights)
        path_pr_list = Common.cal_path_pr(idid)
        entropy_list = Common.cal_entropy(path_pr_list)
        # entropy_list = Common.cal_entropy(Common.get_idid(public_idid, policy_dicts, weights))
        # 2.2、标准化信息熵
        entropy_norm = Common.norm_entropy(entropy_list)
        # 3、多样性和信息熵加权
        fit = Common.cal_fitness(diversity_norm, entropy_norm)
        # 4、更换fitness
        for i in range(len(fit)):
            self.pop[i]["fitness"] = fit[i]

    def get_d(self):
        d_list = list()
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            # ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            d_list.append(fits)
        return d_list

    def genome_to_mat(self, genome):
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            mat[:, cl] = column
        return mat

    def mat_to_pathes(self, mat):
        for cl in range(0, self.dbn.horizon, 1):
            self.tree_template[:, cl * 2] = mat[:, cl]
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        return pathes

    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes

    # ga_main
    def ga_main(self, public_idid):
        # initial v
        self.init_v(len(self.pop), self.geneinfo_dim)
        self.init_p(self.pop)

        # mark the previous time pop
        self.temp_pop = copy.deepcopy(self.pop)
        # 标记出本次迭代最好粒子
        self.bestindividual = Common.get_pbest(copy.deepcopy(self.pop))
        self.pbest = Common.get_pbest(copy.deepcopy(self.pop))

        f_list = [copy.deepcopy(self.pop)[i]['fitness'] for i in range(len(self.pop))]
        self.fits['max'].append(max(f_list))
        self.fits['min'].append(min(f_list))
        self.fits['mean'].append(sum(f_list) / len(f_list))
        self.fits['std'].append(np.std(np.array(f_list)))

        N = self.parameters.values.get('generation_size')
        bar = tqdm(total=N, position=0, desc='PSOC')

        self.CM_num = 0

        for g in range(N):
            self.update_weight(g)
            if not self.progressbar:
                print("-- Generation %i --" % g)
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            nextoff = self.get_pso_pop(selectpop)

            # 本段实现可以放在评估之前，也可放在评估过之后
            """
                1、将每轮迭代的模型代入IDID，求出最有可能是真实模型的n条path
                    用上一次迭代的模型去求解IDID
                2、将迭代后的每个模型与n条path计算距离，得到距离最大的m个模型
                    2.1、计算距离
                        沿用陈老师的距离计算方式进行计算
                    2.2、这m个模型说明突变失败(可能朝着相反的方向运动了，需要重新返回起点进行迭代)
                    2.3、也即是返回到上一个版本(模型一一对应)
                3、用淘汰后的模型和返回到上一个版本的模型进行下一轮迭代
            """

            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, public_idid, final_pop=True)
            else:
                self.next_pop(nextoff, public_idid)
            # 更新self.temp_pop
            self.temp_pop = copy.deepcopy(self.pop)
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            self.pbest = self.selectBest(self.pop)
            if self.pbest['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = copy.deepcopy(self.pbest)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
            bar.update(1)
        # print("纠偏次数: ", self.CM_num)
        bar.close()
        self.result['Fitness'] = self.fits
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()

    def init_v(self, num, dim):
        range_dim = list(np.arange(dim))
        action_dim = list(np.arange(self.dbn.num_as))
        for i in range(num):
            v_list = []
            p = random.sample(range_dim, 1)[0]
            v_list.append(p)
            a = random.sample(action_dim, 1)[0]
            v_list.append(a)
            r = random.randint(1, 10) / 10
            v_list.append(r)
            self.velocity[i] = [v_list]
            self.pop[i]['velocity'] = [v_list]

    def init_p(self, selectpop):
        pop_num = len(selectpop)
        for i in range(pop_num):
            self.position[i] = selectpop[i]['Gene'].data

    def update_weight(self, g):
        LDW_ini = 0.9
        LDW_end = 0.4
        iter = self.parameters.values['generation_size']
        self.weight_pso = ((LDW_ini - LDW_end) * (iter - g) / iter) + LDW_end

    def get_pso_pop(self, selectpop):
        nextoff = []
        pbest_data = self.pbest['Gene'].data
        gbest_data = self.bestindividual['Gene'].data
        for key in range(len(selectpop)):
            p_v_dict = dict()
            # 获取每个粒子的位置
            data = selectpop[key]['Gene'].data.copy()
            w, c1, c2 = self.form_par(self.weight_pso, self.parameters.values['learning_rate1'],
                                      self.parameters.values['learning_rate1'])
            # 通过换位减获取速度
            pop_v0 = self.update_v0(w, selectpop[key]['velocity'])
            pop_v1 = self.update_v_by_p(c1, pbest_data, data)
            pop_v2 = self.update_v_by_p(c2, gbest_data, data)

            update_v = self.pop_v_mul(pop_v0, pop_v1, pop_v2)
            # 更新速度
            # self.velocity[key] = update_v
            # selectpop[key]['velocity'] = update_v
            p_v_dict['velocity'] = update_v
            # 更新粒子位置
            pos_ = self.update_pos(data, update_v)
            # self.position[key] = pos_
            p_v_dict['position'] = pos_
            # selectpop[key]['Gene'].data = pos_
            nextoff.append(p_v_dict)
        off = self.gen_Gene(nextoff)
        return off

    def form_par(self, w, c1, c2):
        # array = np.array([w, c1, c2])
        # r1 = random.random()
        # r2 = random.random()
        array = np.array([w, c1, c2])
        sum = np.sum(array)
        for i in range(len(array)):
            array[i] = array[i] / sum
        return array[0], array[1], array[2]

    def pop_v_mul(self, v1, v2, v3):
        v1_c = v1.copy()
        v2_c = v2.copy()
        v3_c = v3.copy()
        v_add = self.add_v(v1_c, v2_c)
        v = self.add_v(v_add, v3_c)
        return v

    def add_v(self, v1_c, v2_c):
        for i in range(len(v1_c)):
            for j in range(len(v2_c)):
                if v1_c[i][0] == v2_c[j][0]:
                    if v1_c[i][1] == v2_c[j][1]:
                        v2_c[j][2] = v1_c[i][2] + v2_c[j][2]
                        v1_c[i][2] = 100
        v_ = []
        for i in range(len(v1_c)):
            if v1_c[i][2] != 100:
                v_.append(v1_c[i])
        for i in range(len(v2_c)):
            v_.append(v2_c[i])
        for i in v_:
            i[2] = round(i[2], 2)
        return v_

    def update_v0(self, w, v):
        v = copy.deepcopy(v)
        for i in range(len(v)):
            v[i][2] = v[i][2] * w
        return v

    def gen_Gene(self, nextoff):
        popoff = []
        for i in range(len(nextoff)):
            newoff = Gene(data=[])
            newoff.data = nextoff[i]['position']
            v = nextoff[i]['velocity']
            ind = self.gen_id(popoff, self.gen_ind(newoff, v))
            popoff.append(ind)
        return popoff

    def update_pos(self, data, pop_v):
        for key in pop_v:
            if random.random() <= key[2] or key[2] > 1:
                data[key[0]] = key[1]
            else:
                data[key[0]] = random.sample(list(np.arange(self.dbn.num_as)), 1)[0]
        return data

    def update_v_by_p(self, c, best_data, data):
        c = c * random.random()
        list_all = []
        j = 0
        for i in range(len(data)):
            if data[i] == best_data[i]:
                j += 1
                continue
            else:
                v_t = [i, best_data[i], c]
                list_all.append(v_t)
        if j == len(data):
            range_dim = list(np.arange(self.geneinfo_dim))
            action_dim = list(np.arange(self.dbn.num_as))
            list_all.append(
                [random.sample(range_dim, 1)[0], random.sample(action_dim, 1)[0], random.randint(1, 10) / 10])
        return list_all

    # ==================================================================================================
    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd

    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)

        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen

    def selection_group(self, individuals, gp=None):
        return individuals

    def rand_pointer(self):
        if not self.parameters.values.get('weight_mode'):
            pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        else:
            sum_w = np.sum(self.weight)  #
            sum_ = 0
            u = random.random() * sum_w
            for pos in range(0, self.geneinfo_dim):
                sum_ += self.weight[pos]
                if sum_ > u:
                    break
        return pos

    def gen_ind(self, muteoff, v):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        ind['velocity'] = v
        return ind

    def gen_id(self, nextoff, ind, gp=None):
        return ind

    def evaluate_diversity2(self, pop):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(pop))
        for gi in range(0, len(pop)):
            ind = pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity2(popindex, sub_genomes)
        return diversity_pop

    def cal_diversity2(self, popindex, sub_genomes2, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = sub_genomes2.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward2(self, pop):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(pop)):
            ind = pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            pop[gi] = ind
            return [pop[i]['fitness'] for i in range(len(pop))]

    def next_pop(self, nextoff, public_idid, final_pop=None):
        if self.parameters.values.get('pelite_mode'):

            pop_temp = [ind for ind in copy.deepcopy(self.pop)]

            # psoc = PSOC()
            # 计算diversity
            # pre_diversity = Common.evaluate_diversity(self, self.temp_pop)
            # now_diversity = Common.evaluate_diversity(self, self.pop)

            # 计算多样性
            pre_diversity = self.evaluate_diversity2(self.temp_pop)
            now_diversity = self.evaluate_diversity2(nextoff)

            # 计算平均适应度
            pre_reward = self.evaluate_reward2(self.temp_pop)
            now_reward = self.evaluate_reward2(nextoff)

            c = 0
            for i in range(len(pre_reward)):
                if (pre_reward[i] > now_reward[i]):
                    c += 1

            # 判断是否进入种群纠偏
            # if pre_diversity > now_diversity and c >= len(self.pop) / 2:
            #     self.CM_num += 1
            #     # 将上一次的粒子的位置转化为paths
            #     weights, policy_dicts = self.getPathAndWeight2(self.temp_pop)
            #     # 求解IDID
            #     idid = Common.get_idid(public_idid, policy_dicts, weights)
            #     # 获取每个path的概率
            #     path_pr = Common.cal_path_pr(idid)
            #     # 获取path的index和weight
            #     index, weight_m = Common.get_max_pro_path_index(path_pr)
            #     # 获取path和weight
            #     most_path, w_m = Common.get_most_path(policy_dicts, index, weight_m)
            #     # 获取当前模型
            #     weights_now, policy_dicts_now = self.getPathAndWeight2(nextoff)
            #     # 获取当前每个模型与最可能的path的相似度
            #     distance = Common.cal_distance(public_idid, policy_dicts_now, most_path, w_m)
            #     # 获取字典
            #     distance_dict = Common.get_dict(distance)
            #     # 字典排序
            #     new_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
            #     for i in range(3):
            #         num = new_dict[i][0]
            #         C = Common.cal_cross(self.temp_pop[num]['Gene'].data, self.pop[num]['Gene'].data)
            #         # nextoff[num]['Gene'].data = self.temp_pop[num]['Gene'].data
            #         nextoff[num]['Gene'].data = C
            #         nextoff[num]['velocity'] = self.temp_pop[num]['velocity']

            self.pop = [ind for ind in nextoff]
            self.evaluate(public_idid)
            [self.pop.append(ind) for ind in pop_temp]
            if final_pop:
                self.select_nextpop(self.parameters.values.get('tournament_size'))
            else:
                self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate(public_idid)
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

        # self.pop = [ind for ind in nextoff]
        # self.evaluate()

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count == size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]

    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def gen_other(self):  # at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod = len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    for ki in self.result.get(key).keys():
                        if key == 'Fitness':
                            continue
                        if ki >= num_mod:
                            self.result.get(key).pop(ki)
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)

    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                rw, w, p = self.dbn.get_reward(modi=gi)
            else:
                rw, w, p = self.dbn.get_reward()
            # rw, w,p = self.dbn.get_reward()
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights, priors, rewards, policy_dicts

    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of ' + self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf')

    def display_pop(self, pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: ' + key)
                print('>value: ')
                print(ind.get(key))
            print("-----------------")


class CM(object):
    def __init__(self, solver, filepath, result, dstr, num_mod, public_idid):
        self.domain = dstr
        self.type = 'CM'
        # print('Initializing ' + self.type)
        self.num_mod = num_mod
        self.pnames = Name()
        self.solver = solver
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'), solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),
                           expansion_flag=True)
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)

        self.progressbar = True
        self.position = dict()
        self.velocity = dict()
        self.initialize()
        self.initialize_pop(public_idid)
        self.alg_name = ''
        self.plot_flag = False
        self.policyList = None
        self.policyPathWeight = None
        self.temp_pop = None

    def set_alg_name(self, name):
        self.alg_name = name

    def initialize(self):
        self.genomes = dict()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()

    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path

    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path

    def fill_tree_template(self):
        # self.tree_template = np.zeros([self.get_num_path(), self.get_len_path()])
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                # pathes = self.result['policy_dict'].get(key)
                self.tree_template = self.result['policy_dict'].get(key)
                break
            # for i in range(0, self.get_num_path()):
            #     self.tree_template[i, :] = pathes[i]
        # print(self.tree_template)

    def initialize_other(self):
        pass

    def initialize_pop(self, public_idid):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate(public_idid)  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'velocity': 0}

    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()
        self.gen_weight()

    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)

    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for j in range(0, self.get_len_path(), 2):
            hi = int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def pathes_to_mat(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        for i in range(0, self.get_num_path()):
            mat[i, :] = [pathes[i][j] for j in range(0, self.get_len_path(), 2)]
        return mat

    def mat_to_genome(self, mat):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()

        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.get_num_observation(), self.dbn.horizon - cl - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(mat[rw, cl]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome

    def gen_genome_level(self):
        '''
             a	o	a	o	a        a	a  a
             2	0	0	0	2        2  0  2
             2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0 =>[3 2 2 1 1 1 1]
             2	1	2	0	2        2  2  2
             2	1	2	1	0        2  2  0
             '''
        if self.genome_template == None:
            print('============errrrrr')
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.dbn.horizon - cl
            start = start + num
        self.genome_level = genome_level
        # print('-------------------------------')
        # print(genome_level)

    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.dbn.horizon - 1):
            step = np.power(self.dbn.num_os, cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.dbn.num_os)
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
        # print('-------------------------------')
        # print(self.genome_arc)

    def gen_weight(self):
        if not self.parameters.values.get('weight_mode'):
            pass
        w = 1 / self.dbn.horizon
        level = self.genome_level
        self.weight = np.array([w / (level.count(level[i])) for i in range(0, len(level))])
        # print(self.weight)

    def gen_pop(self):
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo), v=0)
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]

    def evaluate(self, public_idid):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(4):
            self.evaluate_entropy(public_idid)
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(5):
            self.evaluate_diversity_entropy(public_idid)

    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind

    def sub_genomes_total_check1(self, tree):
        # optimize it by length
        lt = len(tree)
        if len(self.sub_genomes_total) == 0:
            self.sub_genomes_total['id'] = 0
        if not self.sub_genomes_total.__contains__(lt):
            value = self.sub_genomes_total['id'] + 1
            tree_dict = dict()
            tree_dict[tree] = value
            self.sub_genomes_total[lt] = tree_dict
        else:
            tree_dict = self.sub_genomes_total.get(lt)
            if not tree_dict.__contains__(tree):
                value = self.sub_genomes_total['id'] + 1
                tree_dict[tree] = value
                self.sub_genomes_total[lt] = tree_dict
            else:
                value = tree_dict.get(tree)
        return value

    def sub_genomes_total_check(self, tree):
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value

    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree = str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree

    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward(self):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward()
            self.pop[gi] = ind

    def evaluate_entropy(self, public_idid):
        weights, policy_dicts = self.getPathAndWeight()
        # 获得信息熵
        idid = Common.get_idid(public_idid, policy_dicts, weights)
        path_pr_list = Common.cal_path_pr(idid)
        entropy = Common.cal_entropy(path_pr_list)
        # entropy = Common.cal_entropy(Common.get_idid(public_idid, policy_dicts, weights))
        # entropy = Common.cal_xinxiliang(public_idid, policy_dicts, weights)
        # 替换self.pop中的fitness
        for i in range(len(entropy)):
            self.pop[i]["fitness"] = entropy[i]

    def getPathAndWeight(self):
        weights = dict()
        policy_dicts = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list,
                                     self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            # if self.parameters.values.get('cover_mode'):
            #     rw, w, p = self.dbn.get_reward(modi=gi)
            # else:
            #     rw, w, p = self.dbn.get_reward()
            rw, w, p = self.dbn.get_reward()
            weights[gi] = w
            policy_dicts[gi] = pathes
        return weights, policy_dicts

    def getPathAndWeight2(self, pop):
        weights = dict()
        policy_dicts = dict()
        for gi in range(0, len(pop)):
            ind = pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list,
                                     self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            # if self.parameters.values.get('cover_mode'):
            #     rw, w, p = self.dbn.get_reward(modi=gi)
            # else:
            #     rw, w, p = self.dbn.get_reward()
            rw, w, p = self.dbn.get_reward()
            weights[gi] = w
            policy_dicts[gi] = pathes
        return weights, policy_dicts

    def evaluate_diversity_entropy(self, public_idid):
        # 1、计算diversity
        diversity_list = self.get_d()
        # 1.1、标准化多样性
        diversity_norm = Common.norm_diversity(diversity_list)
        # 2、计算entropy
        weights, policy_dicts = self.getPathAndWeight()
        idid = Common.get_idid(public_idid, policy_dicts, weights)
        path_pr_list = Common.cal_path_pr(idid)
        entropy_list = Common.cal_entropy(path_pr_list)
        # entropy_list = Common.cal_entropy(Common.get_idid(public_idid, policy_dicts, weights))
        # 2.2、标准化信息熵
        entropy_norm = Common.norm_entropy(entropy_list)
        # 3、多样性和信息熵加权
        fit = Common.cal_fitness(diversity_norm, entropy_norm)
        # 4、更换fitness
        for i in range(len(fit)):
            self.pop[i]["fitness"] = fit[i]

    def get_d(self):
        d_list = list()
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(self.pop))
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            # ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex, gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            d_list.append(fits)
        return d_list

    def genome_to_mat(self, genome):
        mat = np.zeros([self.get_num_path(), self.dbn.horizon])
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            mat[:, cl] = column
        return mat

    def mat_to_pathes(self, mat):
        for cl in range(0, self.dbn.horizon, 1):
            self.tree_template[:, cl * 2] = mat[:, cl]
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        return pathes

    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
        ind_start = 0
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes

    # ga_main
    def ga_main(self, public_idid):
        # initial v
        self.init_v(len(self.pop), self.geneinfo_dim)
        self.init_p(self.pop)

        # mark the previous time pop
        self.temp_pop = copy.deepcopy(self.pop)
        # 标记出本次迭代最好粒子
        self.bestindividual = Common.get_pbest(copy.deepcopy(self.pop))
        self.pbest = Common.get_pbest(copy.deepcopy(self.pop))

        f_list = [copy.deepcopy(self.pop)[i]['fitness'] for i in range(len(self.pop))]
        self.fits['max'].append(max(f_list))
        self.fits['min'].append(min(f_list))
        self.fits['mean'].append(sum(f_list) / len(f_list))
        self.fits['std'].append(np.std(np.array(f_list)))

        N = self.parameters.values.get('generation_size')
        bar = tqdm(total=N, position=0, desc='CM')

        self.CM_num = 0

        for g in range(N):
            self.update_weight(g)
            if not self.progressbar:
                print("-- Generation %i --" % g)
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            nextoff = self.get_pso_pop(selectpop)

            # 本段实现可以放在评估之前，也可放在评估过之后
            """
                1、将每轮迭代的模型代入IDID，求出最有可能是真实模型的n条path
                    用上一次迭代的模型去求解IDID
                2、将迭代后的每个模型与n条path计算距离，得到距离最大的m个模型
                    2.1、计算距离
                        沿用陈老师的距离计算方式进行计算
                    2.2、这m个模型说明突变失败(可能朝着相反的方向运动了，需要重新返回起点进行迭代)
                    2.3、也即是返回到上一个版本(模型一一对应)
                3、用淘汰后的模型和返回到上一个版本的模型进行下一轮迭代
            """

            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, public_idid, final_pop=True)
            else:
                self.next_pop(nextoff, public_idid)
            # 更新self.temp_pop
            self.temp_pop = copy.deepcopy(self.pop)
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            self.pbest = self.selectBest(self.pop)
            if self.pbest['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = copy.deepcopy(self.pbest)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
            bar.update(1)
        # print("纠偏次数: ", self.CM_num)
        bar.close()
        self.result['Fitness'] = self.fits
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()

    def init_v(self, num, dim):
        range_dim = list(np.arange(dim))
        action_dim = list(np.arange(self.dbn.num_as))
        for i in range(num):
            v_list = []
            p = random.sample(range_dim, 1)[0]
            v_list.append(p)
            a = random.sample(action_dim, 1)[0]
            v_list.append(a)
            r = random.randint(1, 10) / 10
            v_list.append(r)
            self.velocity[i] = [v_list]
            self.pop[i]['velocity'] = [v_list]

    def init_p(self, selectpop):
        pop_num = len(selectpop)
        for i in range(pop_num):
            self.position[i] = selectpop[i]['Gene'].data

    def update_weight(self, g):
        LDW_ini = 0.9
        LDW_end = 0.4
        iter = self.parameters.values['generation_size']
        self.weight_pso = ((LDW_ini - LDW_end) * (iter - g) / iter) + LDW_end

    def get_pso_pop(self, selectpop):
        nextoff = []
        pbest_data = self.pbest['Gene'].data
        gbest_data = self.bestindividual['Gene'].data
        for key in range(len(selectpop)):
            p_v_dict = dict()
            # 获取每个粒子的位置
            data = selectpop[key]['Gene'].data.copy()
            w, c1, c2 = self.form_par(self.weight_pso, self.parameters.values['learning_rate1'],
                                      self.parameters.values['learning_rate1'])
            # 通过换位减获取速度
            pop_v0 = self.update_v0(w, selectpop[key]['velocity'])
            pop_v1 = self.update_v_by_p(c1, pbest_data, data)
            pop_v2 = self.update_v_by_p(c2, gbest_data, data)

            update_v = self.pop_v_mul(pop_v0, pop_v1, pop_v2)
            # 更新速度
            # self.velocity[key] = update_v
            # selectpop[key]['velocity'] = update_v
            p_v_dict['velocity'] = update_v
            # 更新粒子位置
            pos_ = self.update_pos(data, update_v)
            # self.position[key] = pos_
            p_v_dict['position'] = pos_
            # selectpop[key]['Gene'].data = pos_
            nextoff.append(p_v_dict)
        off = self.gen_Gene(nextoff)
        return off

    def form_par(self, w, c1, c2):
        # array = np.array([w, c1, c2])
        # r1 = random.random()
        # r2 = random.random()
        array = np.array([w, c1, c2])
        sum = np.sum(array)
        for i in range(len(array)):
            array[i] = array[i] / sum
        return array[0], array[1], array[2]

    def pop_v_mul(self, v1, v2, v3):
        v1_c = v1.copy()
        v2_c = v2.copy()
        v3_c = v3.copy()
        v_add = self.add_v(v1_c, v2_c)
        v = self.add_v(v_add, v3_c)
        return v

    def add_v(self, v1_c, v2_c):
        for i in range(len(v1_c)):
            for j in range(len(v2_c)):
                if v1_c[i][0] == v2_c[j][0]:
                    if v1_c[i][1] == v2_c[j][1]:
                        v2_c[j][2] = v1_c[i][2] + v2_c[j][2]
                        v1_c[i][2] = 100
        v_ = []
        for i in range(len(v1_c)):
            if v1_c[i][2] != 100:
                v_.append(v1_c[i])
        for i in range(len(v2_c)):
            v_.append(v2_c[i])
        for i in v_:
            i[2] = round(i[2], 2)
        return v_

    def update_v0(self, w, v):
        v = copy.deepcopy(v)
        for i in range(len(v)):
            v[i][2] = v[i][2] * w
        return v

    def gen_Gene(self, nextoff):
        popoff = []
        for i in range(len(nextoff)):
            newoff = Gene(data=[])
            newoff.data = nextoff[i]['position']
            v = nextoff[i]['velocity']
            ind = self.gen_id(popoff, self.gen_ind(newoff, v))
            popoff.append(ind)
        return popoff

    def update_pos(self, data, pop_v):
        for key in pop_v:
            if random.random() <= key[2] or key[2] > 1:
                data[key[0]] = key[1]
            else:
                data[key[0]] = random.sample(list(np.arange(self.dbn.num_as)), 1)[0]
        return data

    def update_v_by_p(self, c, best_data, data):
        c = c * random.random()
        list_all = []
        j = 0
        for i in range(len(data)):
            if data[i] == best_data[i]:
                j += 1
                continue
            else:
                v_t = [i, best_data[i], c]
                list_all.append(v_t)
        if j == len(data):
            range_dim = list(np.arange(self.geneinfo_dim))
            action_dim = list(np.arange(self.dbn.num_as))
            list_all.append(
                [random.sample(range_dim, 1)[0], random.sample(action_dim, 1)[0], random.randint(1, 10) / 10])
        return list_all

    # ==================================================================================================
    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd

    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)

        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen

    def selection_group(self, individuals, gp=None):
        return individuals

    def rand_pointer(self):
        if not self.parameters.values.get('weight_mode'):
            pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        else:
            sum_w = np.sum(self.weight)  #
            sum_ = 0
            u = random.random() * sum_w
            for pos in range(0, self.geneinfo_dim):
                sum_ += self.weight[pos]
                if sum_ > u:
                    break
        return pos

    def gen_ind(self, muteoff, v):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        ind['velocity'] = v
        return ind

    def gen_id(self, nextoff, ind, gp=None):
        return ind

    def evaluate_diversity2(self, pop):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        sub_genomes = dict()
        self.sub_genomes_total = dict()
        popindex = range(0, len(pop))
        for gi in range(0, len(pop)):
            ind = pop[gi]
            gen = ind['Gene'].data
            subtree = self.gen_genome_subtree(gen)
            sub_genomes[gi] = subtree
        diversity_pop = self.cal_diversity2(popindex, sub_genomes)
        return diversity_pop

    def cal_diversity2(self, popindex, sub_genomes2, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = sub_genomes2.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity

    def evaluate_reward2(self, pop):
        # evaluate the reward of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(pop)):
            ind = pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            pop[gi] = ind
            return [pop[i]['fitness'] for i in range(len(pop))]

    def next_pop(self, nextoff, public_idid, final_pop=None):
        if self.parameters.values.get('pelite_mode'):

            pop_temp = [ind for ind in copy.deepcopy(self.pop)]

            # psoc = PSOC()
            # 计算diversity
            # pre_diversity = Common.evaluate_diversity(self, self.temp_pop)
            # now_diversity = Common.evaluate_diversity(self, self.pop)

            # 计算多样性
            pre_diversity = self.evaluate_diversity2(self.temp_pop)
            now_diversity = self.evaluate_diversity2(nextoff)

            # 计算平均适应度
            pre_reward = self.evaluate_reward2(self.temp_pop)
            now_reward = self.evaluate_reward2(nextoff)

            c = 0
            for i in range(len(pre_reward)):
                if (pre_reward[i] > now_reward[i]):
                    c += 1

            # 判断是否进入种群纠偏
            # if pre_diversity > now_diversity and c >= len(self.pop) / 2:
            #     self.CM_num += 1
            #     # 将上一次的粒子的位置转化为paths
            #     weights, policy_dicts = self.getPathAndWeight2(self.temp_pop)
            #     # 求解IDID
            #     idid = Common.get_idid(public_idid, policy_dicts, weights)
            #     # 获取每个path的概率
            #     path_pr = Common.cal_path_pr(idid)
            #     # 获取path的index和weight
            #     index, weight_m = Common.get_max_pro_path_index(path_pr)
            #     # 获取path和weight
            #     most_path, w_m = Common.get_most_path(policy_dicts, index, weight_m)
            #     # 获取当前模型
            #     weights_now, policy_dicts_now = self.getPathAndWeight2(nextoff)
            #     # 获取当前每个模型与最可能的path的相似度
            #     distance = Common.cal_distance(public_idid, policy_dicts_now, most_path, w_m)
            #     # 获取字典
            #     distance_dict = Common.get_dict(distance)
            #     # 字典排序
            #     new_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
            #     for i in range(3):
            #         num = new_dict[i][0]
            #         # C = Common.cal_cross(self.temp_pop[num]['Gene'].data, self.pop[num]['Gene'].data)
            #         nextoff[num]['Gene'].data = self.temp_pop[num]['Gene'].data
            #         # nextoff[num]['Gene'].data = C
            #         nextoff[num]['velocity'] = self.temp_pop[num]['velocity']

            self.pop = [ind for ind in nextoff]
            self.evaluate(public_idid)
            [self.pop.append(ind) for ind in pop_temp]
            if final_pop:
                self.select_nextpop(self.parameters.values.get('tournament_size'))
            else:
                self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate(public_idid)
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))

        # self.pop = [ind for ind in nextoff]
        # self.evaluate()

    def select_nextpop(self, size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count == size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]

    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def gen_other(self):  # at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod = len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    for ki in self.result.get(key).keys():
                        if key == 'Fitness':
                            continue
                        if ki >= num_mod:
                            self.result.get(key).pop(ki)
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree(self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The Policy Tree of ' + self.type + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer'), self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)

    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of ' + self.type, self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step, expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                rw, w, p = self.dbn.get_reward(modi=gi)
            else:
                rw, w, p = self.dbn.get_reward()
            # rw, w,p = self.dbn.get_reward()
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights, priors, rewards, policy_dicts

    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of ' + self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf')

    def display_pop(self, pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: ' + key)
                print('>value: ')
                print(ind.get(key))
            print("-----------------")