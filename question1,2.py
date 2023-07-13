#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


# # Problem 1

def PSO(func, x_min, x_max, d, v_max=None, w=0.7, a1=2, a2=2, n=20, repulsion=False, a3=1):
    # initialization
    x = x_min + (x_max-x_min)*np.random.rand(n, d)
    p = np.zeros((n, d))
    v = np.zeros((n, d))
    fg = np.inf
    fp = np.inf * np.ones(n)
    path = []
    if v_max == None:
        v_max = 0.1 * (x_max-x_min)
    
    n_no_change = 0
    iters = 0
    min_iter = 10000
    while n_no_change < 50 or iters < min_iter:
        fg_start = fg
        for i in range(n):
            fi = func(x[i])
            if fi < fp[i]:
                p[i] = x[i]
                fp[i] = fi
            if fi < fg:
                g = x[i]
                fg = fi
                path.append(x[i])
        
        v = w*v + a1*np.random.rand(n, 1)*(p-x) + a2*np.random.rand(n, 1)*(g-x)
        if repulsion:
            v += a3 * get_repulsion(x)
        v = np.clip(v, -v_max, v_max)
        x = x + v
        x = np.clip(x, x_min, x_max)
        
        if fg == fg_start:
            n_no_change += 1
        else:
            n_no_change = 0
        iters += 1
    return g, fg, path

def get_repulsion(x):
    Z = np.zeros_like(x)
    for i in range(x.shape[0]):
        Z[i] = (np.random.rand(x.shape[0],1)*(x[i] - x) / (((x[i] - x)**2).sum(axis=1)+1e-8).reshape(-1, 1)).sum(axis=0)
    return Z


# 1.1

def f1(x):
    return (x**2).sum()


round_vec = np.vectorize(round)
g, fg, path = PSO(f1, x_min=-5.12, x_max=5.12, d=3)
print(round_vec(g, 5), round(fg,5))


def f2(x):
    d = x.shape[0]
    return 10*d + (x**2).sum() -10*np.cos(2*np.pi*x).sum()


g, fg, path = PSO(f2, x_min=-5.12, x_max=5.12, d=3, a1=4, a2=1)
print(round_vec(g, 5), round(fg,5))


g, fg, path = PSO(f2, x_min=-5.12, x_max=5.12, d=10, a1=4, a2=1)
print(round_vec(g, 5), round(fg,5))


# 1.3

g, fg, path = PSO(f2, x_min=-5.12, x_max=5.12, d=3, a1=4, a2=1, repulsion=True)
print(round_vec(g, 5), round(fg,5))


g, fg, path = PSO(f2, x_min=-5.12, x_max=5.12, d=10, a1=4, a2=1, repulsion=True, a3=0.2)
print(round_vec(g, 5), round(fg,5))


g, fg, path = PSO(f1, x_min=-5.12, x_max=5.12, d=3, repulsion=True, a3=1)
print(round_vec(g, 5), round(fg,5))


# # Problem 2 & 4

x = -5.12+10.24*np.random.rand(100)
y = -5.12+10.24*np.random.rand(100)
data1 = []
data2 = []
for i in range(100):
    data1.append((x[i], y[i], f1(np.array([x[i],y[i]]))))
    data2.append((x[i], y[i], f2(np.array([x[i],y[i]]))))


from random import sample, random, randint, choice
from copy import deepcopy
import math

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


pi = math.pi
e = math.e
pset = {'+':2, '*':2}
v = ['x', 'y']
for item in v:
    pset[item] = 0
s = primitive_handler(pset, v)

data = data1

popsize = 100
max_depth = 3
cross_rate = 0.90
rep_rate = 0.98
mut_rate = 0.99
shrink_rate = 1
tourn_size = 7
target_fitness = 0.1


pop = []
#### half and half
for _ in range(popsize//2):
    pop.append(BinaryTree(s, 'full', randint(1, max_depth)))
for _ in range(popsize-popsize//2):
    pop.append(BinaryTree(s, 'grow', randint(1, max_depth)))
    

solution = evolve(pop)
winner = deepcopy(solution['best'])
print('The winning program is', solution['program'])
print('Its fitness score is', solution['score'], 'and it appears in generation', solution['gen'])


def cos2pix(x):
    return np.cos(2*np.pi*x)
pset = {'+':2, '-':2, '*':2, '20':0, '10':0, '2':0, 'cos2pix':1}
v = ['x', 'y']
for item in v:
    pset[item] = 0
s = primitive_handler(pset, v)
data = data2

popsize = 50
max_depth = 4
cross_rate = 0.80
rep_rate = 0.85
mut_rate = 0.9
pointmut_rate = 0.95
shrink_rate = 1
tourn_size = 10
target_fitness = 0.1


pop = []
#### half and half
for _ in range(popsize//2):
    pop.append(BinaryTree(s, 'full', randint(1, max_depth)))
for _ in range(popsize-popsize//2):
    pop.append(BinaryTree(s, 'grow', randint(1, max_depth)))

solution = evolve(pop, max_generation=30, min_nodes=0, localsearch=True)
winner = deepcopy(solution['best'])
print('The winning program is', solution['program'])
print('Its fitness score is', solution['score'], 'and it appears in generation', solution['gen'])




