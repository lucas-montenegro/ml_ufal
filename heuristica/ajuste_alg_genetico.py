# -*- coding: utf-8 -*-
import random as rand
import requests as req
import numpy as np
import json
from matplotlib import pyplot as plt


MUTATION_CHANCE = .15
MUTATION_RANGE = 45

POPULATION_SIZE = 50
INDIVIDUALS_TO_KEEP = 4
CRITERION_TO_CHANGE_SELECTION = .8

GENERATIONS = 50

RUNNING_TIMES = 50  # quantas vezes o algoritmo será testado

TEST_URL = 'http://localhost:8080/antenna/simulate?phi1={0}&theta1={3}&phi2={1}&theta2={4}&phi3={2}&theta3={5}'
URL = 'https://aydanomachado.com/mlclass/02_Optimization.php?phi1={0}&theta1={3}&phi2={1}&theta2={4}&phi3={2}&theta3={5}&dev_key=LW'


class Antena(object):
    def __init__(self, genoma):
        self.genoma = genoma
        self.fitness_value = None
        self.fitness_calc()

    @staticmethod
    def create():
        genoma = [rand.randint(0, 360) for i in range(6)]
        return Antena(genoma)

    @staticmethod
    def cross(a, b):
        pos = rand.randint(1, 5)
        new = a.genoma[:pos] + b.genoma[pos:]
        new_antena = Antena(new)
        new_antena.mutate()
        return new_antena

    @staticmethod
    def sort_key(a):
        return a.fitness_value

    def _test_fitness_calc(self):
        r = req.get(TEST_URL.format(*self.genoma))
        self.fitness_value = float(r.text.split('\n')[0])

    def _fitness_calc(self):
        r = req.get(URL.format(*self.genoma))
        self.fitness_value = json.loads(r.text)['gain']

    def fitness_calc(self):
       self._test_fitness_calc() 
       #self._fitness_calc()

    def mutate(self):
        gene_mutate_chance = MUTATION_CHANCE*100
        for i in range(len(self.genoma)):
            if self._will_mutate(gene_mutate_chance):
                self.genoma[i] += rand.randint(-MUTATION_RANGE, MUTATION_RANGE+1)
                if self.genoma[i] < 0: self.genoma[i] = 0
                if self.genoma[i] > 359: self.genoma[i] = 359

    def _will_mutate(self, prob) -> bool:
        return rand.randint(0, 101) <= prob

    def __str__(self):
        out = f'{self.fitness_value}\nPhi    Theta\n'
        for i in range(3):
            out += f'{self.genoma[i]:>3}     {self.genoma[i+3]:>3}\n'
        return out


class Population(object):
    def __init__(self, size):
        self.size = size
        self.generation = 0
        self.individuals = []
        self.selection_weights_list = []
        self._start()

    def _start(self):
        for i in range(self.size):
            a = Antena.create()
            self.individuals.append(a)
        self.individuals = sorted(self.individuals, key=Antena.sort_key, reverse=True)

    def __calc_selection_weights_method1(self):
        fitness = [i.fitness_value+100 for i in self.individuals]
        total = sum(fitness)
        self.selection_weights_list = [i/total for i in fitness]            
    
    def __calc_selection_weights_method2(self):
        weights = np.arange(POPULATION_SIZE, 0, -1)
        total = sum(weights)
        self.selection_weights_list = [weights[i]/total for i in range(len(weights))]
        
    def _calc_selection_weights(self, method=1):
        if method == 1: 
            self.__calc_selection_weights_method1()
        else:
            self.__calc_selection_weights_method2()      

    def generate_new_population(self, selection_method=1):
        self._calc_selection_weights(selection_method)
        new_pop = []
        for i in range(self.size - INDIVIDUALS_TO_KEEP):
            a, b = np.random.choice(self.individuals, 2, False, 
                                    self.selection_weights_list)
            new_pop.append(Antena.cross(a, b))
        new_pop += self.individuals[:INDIVIDUALS_TO_KEEP]
        self.individuals = sorted(new_pop, key=Antena.sort_key, reverse=True)
        self.generation += 1


def plot(bests, kind='box'):
    if kind == 'box':
        plt.boxplot(bests)
    else:
        plt.plot(range(len(bests)), bests)
        plt.xticks(range(1, GENERATIONS+2, 2), fontsize=8)
    plt.yticks(range(22,32), fontsize=8)
    plt.tight_layout()
    plt.show()
    

def print_statistics(fitness_list):
    fl = np.array(sorted(fitness_list, reverse=True))
    np.sort(fl)
    print(f'Mean: {fl.mean():.6f}   Std: {fl.std():.6f}')
    print(f'Best: {fl[0]:.6f}  Worst: {fl[-1]:.6f}')


def start_new_generation(pop):
    selection_method = 1
    if pop.generation >= GENERATIONS * CRITERION_TO_CHANGE_SELECTION:
        selection_method = 2
    pop.generate_new_population(selection_method)


def run():
    pop = Population(POPULATION_SIZE)
    bests = []
    for i in range(GENERATIONS):
        best = pop.individuals[0]
        bests.append(best.fitness_value)
        if i != 0:
           start_new_generation(pop)
           #pop.generate_new_population()
    return best, bests


global_best_fitnesses, global_bests = [], []
def test():
    global global_bests
    for i in range(RUNNING_TIMES):
        best, on_run_bests = run()
        global_best_fitnesses.append(best.fitness_value)
        global_bests.append(best)
    #print(best.fitness_value,'\n', best.genoma)
    plot(global_best_fitnesses)
    global_bests = sorted(global_bests, key=Antena.sort_key, reverse=True)
    print_statistics(global_best_fitnesses)
    

def main():
    best, bests = run()
    plot(bests, 'line')
    print_statistics(bests)
    print(best.genoma)

#test()
main()

