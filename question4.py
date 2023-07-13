# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 00:35:23 2021

"""

from random import sample, random, randint, choice
from copy import deepcopy
import math
import numpy as np

popsize = 100
max_depth = 4
cross_rate = 0.80
rep_rate = 0.85
mut_rate = 0.9
pointmut_rate = 0.95
shrink_rate = 1
tourn_size = 10
target_fitness = 0.1

def f1(x):
    return np.power(x, 2)
def f2(x):
    return np.power(x, 3)
def f3(x):
    return np.power(x, 4)
def f4(x):
    return np.power(x, 1/2)
def f5(x):
    return np.power(x, 1/3)
def f6(x):
    return np.power(x, 1/4)



class Node(object):
    def __init__(self, value, arity):
        self.value = value
        self.arity = arity



class BinaryTree(list):
    def __init__(self, s, method, depth=None):
        self.primitives = s['primitives']  #e.g. ['+', '**', 'x', 'y']
        self.set_dict = s 
        self.depth = depth
        self.size = 2 ** (self.depth + 1) - 1   # num_nodes
        self.extend([None]*self.size)  # empty complete tree
        self.last_level = 2 ** self.depth - 1  # index of 1st node in the last level
        if method == 'full':
            self._full(self.size, self.last_level, 0)
        if method == 'grow':
            self._grow(self.size, self.last_level, 0)
            
    def draw(self):
        print('  ',self[0].value)
        print('',self[1].value, ' ', self[2].value)
        if self.depth >= 2:
            print(self[3].value, self[4].value, self[5].value, self[6].value)
        if self.depth >= 3:
            print(self[7].value, self[8].value, self[9].value, self[10].value, self[11].value, self[12].value, self[13].value, self[14].value)
    
    def has_children(self, n):  # True = has at least one child
        if (2 * n + 1) >= len(self) or (self[2*n+1] == None and self[self.get_right_index(n)] == None):
            return False
        else:
            return True

    def _full(self, s, m, n):
        if n < m: # n is not on the last level
            prim = choice(self.set_dict['functions'])
            self[n] = Node(prim, self.set_dict['dict'][prim])  # internal node
            self._full(s, m, 2*n+1) # left child
            self._full(s, m, 2*n+2) # right child
        elif (n < s):
            self[n] = Node(choice(self.set_dict['terminals']), 0)  # terminal
            
    def _grow(self, s, m, n):
        if n > 0:
            parent = self[int((n-1)/2)]
        if n == 0: 
            if self.depth >= 1:
                prim = choice(self.set_dict['primitives'])
            elif self.depth == 0:
                prim = choice(self.set_dict['terminals'])
            self[n] = Node(prim, self.set_dict['dict'][prim])
            self._grow(s, m, 2*n+1)
            self._grow(s, m, 2*n+2)
        elif n < m:
            if parent == None or parent.arity == 0:
                self[n] = None
            else:
                prim = choice(self.set_dict['primitives'])
                self[n] = Node(prim, self.set_dict['dict'][prim])
            self._grow(s, m, 2*n+1)
            self._grow(s, m, 2*n+2)
        elif n < s:
            if parent == None or parent.arity == 0:
                self[n] = None
            else:
                self[n] = Node(choice(self.set_dict['terminals']), 0)
    
    def build_program(self, n=0):
        string = ''
        if n < self.size and self[n] != None:
            string = self[n].value
            if self[n].arity != 1:
                left = self.build_program(2*n+1)
                right = self.build_program(2*n+2)
                string = '(' + left + string + right + ')'
            if self[n].arity == 1:
                left = self.build_program(2*n+1)
                string = '(' + string +  left  + ')'
        return string

    def get_rand_terminal(self):
        index = randint(0, self.size-1)
        if (self[index] == None) or (self[index].arity > 0):
            return self.get_rand_terminal()
        return index

    def get_rand_function(self):
        index = randint(0, self.last_level-1)
        if (self[index] == None) or (self[index].arity == 0):
            return self.get_rand_function()
        return index

    def get_rand_node(self):
        index = randint(0, self.size-1)
        if self[index] == None:
            return self.get_rand_node()
        return index
        
    def get_subtree(self, n, depth=0):
        if n >= len(self):
            return []
        start = n
        stop = (2 ** depth) + n
        subtree = self[start:stop] # node
        return subtree + self.get_subtree(2*start+1, depth+1)  # list of Nodes but not a BinaryTree object

    def _fill_subtree(self, n, subtree, depth=0):   #subtree: list of Nodes
        if n >= self.size:
            return
        start = n
        stop = (2 ** depth) + n
        for i in range(start,stop):
            self[i] = subtree.pop(0)
        self._fill_subtree(2*start+1, subtree, depth+1)

    def _pad(self, n, subtree):  
        old = self.get_subtree(n)
        new = subtree
        nodes_in_old = len(old)
        nodes_in_new = len(new)
        if nodes_in_new == nodes_in_old:
            return
        if nodes_in_new < nodes_in_old:
            new.extend([None]*(int(next_level_size(nodes_in_new))))
        elif nodes_in_new > nodes_in_old:
            self.extend([None]*(int(next_level_size(self.size))))
            self.size = len(self)
        self._pad(n, new)

    def replace_subtree(self, n, subtree):
        self._pad(n, subtree)
        self._fill_subtree(n, subtree)
        if len(self) > self.size:
            del self[self.size:]

def get_depth(k):
    return int(np.log2(k + 1) - 1)

def next_level_size(k):
    d = get_depth(k) + 1
    return 2 ** d


def subtree_crossover(population, n, data):
    exception_occurred = False
    parent1 = tournament(population, n, data)[0]
    parent2 = tournament(population, n, data)[0]
    cross_pt1 = parent1.get_rand_node()
    cross_pt2 = parent2.get_rand_node()
    return _crossover(parent1, parent2, cross_pt1, cross_pt2)  # a tree object

def subtree_mutation(tree, max_depth):   # equivalent to crossover with a random tree
    s = tree.set_dict
    methods = ['full', 'grow']
    tmp_tree = BinaryTree(s, choice(methods), randint(0, max_depth))
    return _crossover(tree, tmp_tree, tree.get_rand_node(), 0)

def point_mutation(tree):
    treecopy = deepcopy(tree)
    if treecopy.size > 1:
        index = treecopy.get_rand_function()
        treecopy[index].value = choice(tree.set_dict['functions'])
        treecopy[index].arity = tree.set_dict['dict'][treecopy[index].value]
    return treecopy
    
def subtree_shrink(tree):
    treecopy = deepcopy(tree)
    if treecopy.size > 1:
        index = treecopy.get_rand_function()
        treecopy.replace_subtree(index, [Node(choice(tree.set_dict['terminals']), 0)])
    return treecopy

def reproduction(population, n, data):
    winner = tournament(population, n, data)[0]
    return deepcopy(winner)   # a tree


def fitness(tree, data, min_nodes=0):
    if len([node for node in tree if node!= None])<min_nodes:
        return np.inf
    prog = tree.build_program()
    variables = tree.set_dict['variables']
    m = len(variables)
    total_err = 0
    for item in data:
        for i in range(m):
            vars()[variables[i]] = item[i]   #assign data value to vars()
        try:
            y_true = item[-1]
            y_eval = eval(prog)
            total_err += (y_true - y_eval) ** 2
        except:   #  illegal programs or problems like 1/0
            return np.inf
    return total_err

def tournament(population, n, data, min_nodes=0):
    pop_sample = sample(population, n)
    best = None
    best_score = np.inf
    for tree in pop_sample:
        score = fitness(tree, data, min_nodes)
        if score < best_score:
            best = tree
            best_score = score

    if best == None:  # all candidates are illegal or have other problems
        return tournament(population, n, data, min_nodes)
    return best, best_score   # a tree, score

def _crossover(tree1, tree2, cross_pt1, cross_pt2):
    tree1copy = deepcopy(tree1)
    tree2copy = deepcopy(tree2)
    sub = tree2copy.get_subtree(cross_pt2)
    tree1copy.replace_subtree(cross_pt1, sub)
    return tree1copy




def primitive_handler(prim_dict, variables):
    functions = []
    terminals = []
    for key in prim_dict:
        if prim_dict[key] == 0:
            terminals.append(key)
        else:
            functions.append(key)
    primitives = functions + terminals
    return {'primitives':primitives, 'functions':functions,
            'terminals':terminals, 'variables':variables, 'dict':prim_dict}

def local_search(pop, data, terminals):
    pop_ = deepcopy(pop)
    for i,tree in enumerate(pop):
        score = fitness(pop_[i], data)
        for j,node in enumerate(tree):
            if node == None:
                continue
            if node.arity == 0:
                for s in terminals:
                    tree[j].value = s
                    new_score = fitness(tree, data)
                    if new_score < score:
                        score = new_score
                        pop_[i][j].value = s
                tree[j].value = pop_[i][j].value
    return pop_
                        
def evolve(pop, generation=1, max_generation=30, min_nodes=0, localsearch=False):  
    if localsearch:
        pop = local_search(pop, data, s['terminals'])
    best, score = tournament(pop, len(pop), data, min_nodes)
    print('gen', generation, 'best score:', round(score))
    if score < target_fitness or generation >= max_generation:
        return {'best':best, 'score':score, 'gen': generation, 'program': best.build_program()}
    next_gen = []
    for i in range(len(pop)):
        try:
            p = random()
            if p < cross_rate:
                child = subtree_crossover(pop, tourn_size, data)
            elif p < rep_rate:
                child = reproduction(pop, tourn_size, data)
            elif p < mut_rate:
                child = subtree_mutation(pop[i], max_depth)
            elif p < pointmut_rate:
                child = point_mutation(pop[i])
            elif p < shrink_rate:
                child = subtree_shrink(pop[i])
        except:
            child = deepcopy(pop[i])
        # trim and rectify 
        if get_depth(len(child)) >= max_depth:
            del child[2**(max_depth+1)-1:]
        child.depth = get_depth(len(child))
        child.size = 2 ** (child.depth + 1) - 1  
        child.last_level = 2 ** child.depth - 1 
        next_gen.append(child)
    return evolve(next_gen, generation+1, max_generation, min_nodes, localsearch)

filename='datafile.txt'
data = []
file = open(filename, 'r')
for line in file:
    line_string = line.rstrip('\n')
    line_list = line_string.split(' ')
    for i in range(len(line_list)):
        line_list[i] = float(line_list[i])
    line_tuple = tuple(line_list)
    data.append(line_tuple)
file.close()

pset = {'+':2, '-':2, '*':2, '/':2,
        '100':0, '10':0, '4':0, '3':0, '2':0, '1':0, 'np.pi':0, 
        'np.sin':1, 'np.cos':1, 'np.exp':1, 'abs':1, 
        'f1':1, 'f2':1, 'f3':1, 'f4':1, 'f5':1, 'f6':1, }
#pset = {'+':2, '*':2}
v = ['x', 'y']
for item in v:
    pset[item] = 0
s = primitive_handler(pset, v)

pop = []
#### half and half
for _ in range(popsize//2):
    pop.append(BinaryTree(s, 'full', randint(1, max_depth)))
for _ in range(popsize-popsize//2):
    pop.append(BinaryTree(s, 'grow', randint(1, max_depth)))

solution = evolve(pop, max_generation=30, min_nodes=0, localsearch=True)
#winner = deepcopy(solution['best'])
#print('The winning program is', solution['program'])
#print('Its fitness score is', solution['score'], 'and it appears in generation', solution['gen'])
print(solution['program'])