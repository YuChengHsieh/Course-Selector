# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:34:39 2021

@author: user
"""
import pandas as pd
import numpy as np
import random


#preprocessing
GA_POPULATION_SIZE = 100
GA_GENERATION = 20
GA_CROSSOVER_RATE = 0.9
GA_MUTATION_RATE = 0.05
GA_K_TOURNAMENT = 5

#set the condition of user input
class condition:
    def __init__(self,department=None,GPA=None,grade=None,credit_upperbound=None,
                 credit_lowerbound=None,require=None,general=None):
        self.department=department
        self.GPA = GPA
        self.grade = grade
        self.credit_upperbound = credit_upperbound
        self.credit_lowerbound = credit_lowerbound
        self.require = require
        self.general = general
        
#interface for user to input conditions
def interface(Course_number):
    department = input('請輸入你的科系 <PME/ESS>:\n')
    grade = int(input('請輸入你的年級 <1~6>:\n'))
    GPA = float(input('請輸入你預期的GPA至少要到多少 <0~4.3>:\n'))
    credit = input('請輸入你預計修習的學分範圍 (ex:18~25請輸入18 25):\n').split()
    require = input('請先自行輸入你的必修以及一定想修的課程之課號 \n(如有兩堂以上請用空白鍵做區隔)\n如果沒有請輸入N:\n').split()
    if require[0] != 'N':
        position = 0
        while position < len(require):
            while require[position] not in Course_number:
                print('目前科號:',require)
                temp = input('科號' + require[position] + '不存在，請問是否要修正此科號?\n如要修正，請重新輸入正確科號'\
                             '如不修正，請輸入N:\n')
                if temp == 'N':
                    require.pop(position)
                    position -= 1
                    break
                else:
                    require[position] = temp
            position += 1
    if len(require) == 0:
        require.append('N')
    general = input('請輸入通識的向度或領域(如果同一個領域要修兩堂或是有不同領域則以空格替代):\n'\
                    '向度直接填數字，領域則填(自、社、人)'\
                    'ex:修兩堂向度1的課輸入 1 1\n如果沒有則輸入N\n').split()
    if general == 'N':
        for i in range(len(general)):
            if general[i] == '自':
                general[i] = 'Science'
            elif general[i] == '人':
                general[i] = 'Humanity'
            else:
                general[i] = 'Society'
    my_condition = condition(department=department,GPA=GPA,grade=grade,
                             credit_upperbound=int(credit[1]),credit_lowerbound=int(credit[0]),
                             require=require,general=general)
    return my_condition    
            
        
#lesson_required is to get the required lesson
def lesson_required(require,lesson) -> tuple:
    total_class = 0
    total_credit = 0
    total_GPA = 0
    curriculum = pd.DataFrame(np.zeros((13,6)))
    index = ['1','2','3','4','n','5','6','7','8','9','a','b','c']
    curriculum.columns = ['Mon','Tue','Wen','Thr','Fri','Else']
    curriculum.index = index
    if require[0] != 'N':
        cmp = {'M':'Mon','T':'Tue','W':'Wen','R':'Thr','F':'Fri'}
        for i in require:
            total_class += 1
            required_lesson = lesson[lesson['Course No.'] == i].reset_index(drop=True)
            index_count = 0
            if required_lesson['Time'][0] == 'ALL':
                total_GPA += required_lesson['Average GPA'][0]
                total_credit = required_lesson['Credit'][0]
                curriculum['Else'][index[index_count]] = required_lesson['Course Title'][0]
            else:
                total_GPA += required_lesson['Average GPA'][0]
                total_credit = required_lesson['Credit'][0]
                required_lesson['Time'] = required_lesson['Time'].str.split('')
                time = list(filter(lambda a : a != '',required_lesson['Time'][0]))
                day = ''
                for j in time:
                    if j.isnumeric() or j in {'n','a','b','c'}:
                       curriculum.loc[j,day] = required_lesson['Course Title'][0]
                    else:
                        day = cmp.get(j)
            
    return curriculum, total_class, total_GPA, total_credit

def output(dataset,name):
    dataset.to_csv(f'{name}.csv',encoding='utf_8_sig',index=None)

#### Chromosome Class ####
class Chromosome:
    def __init__(self, dim) -> None:
        self.gene = [random.randint(0, 1) for i in range(dim)]
        self.fitness = 0.0
        self.GPA = 0
        self.credit = 0
        self.curr = None

    @classmethod
    def crossover(cls, parent1, parent2, xover_rate) -> tuple:

        dim = len(parent1.gene)
        child1, child2 = Chromosome(dim), Chromosome(dim)

        if random.random() < xover_rate:
            for i in range(dim):
                if random.random() < 0.5:
                    child1.gene[i] = parent2.gene[i]
                    child2.gene[i] = parent1.gene[i]
                else:
                    child1.gene[i] = parent1.gene[i]
                    child2.gene[i] = parent2.gene[i]

        return child1, child2

    @classmethod
    def mutate(cls, chrm, mutate_rate) -> None:

        dim = len(chrm.gene)
        for i in range(dim):
            if random.random() < mutate_rate:
                if(chrm.gene[i] == 0):
                    chrm.gene[i] = 1
                else:
                    chrm.gene[i] = 0
        return
    def print(self) -> None:
        print(self.curr)
        print("Averange GPA: " + str(round(self.GPA, 2)))
        print("Total credit: " + str(self.credit))


class GA:
    def __init__(self, condition, major_selected, GE_selected, curriculum, pop_size, k,
                 xover_rate, mutate_rate , base_total_GPA, base_total_credit, base_total_class):

        #### Validate arguments ####
        if (not isinstance(pop_size, int)) or (pop_size < 1):
            raise ValueError("`pop_size` can only be positive integer.")

        if (not isinstance(k, int)) or (k < 1) or (k > pop_size):
            raise ValueError("`k` for tournament selection can only an integer between 1 and `pop_size`.")

        #### Variable declaration ####
        self.chrm_len = len(major_selected.index) + len(GE_selected.index)
        self.condition = condition
        self.major = major_selected
        self.GE = GE_selected
        self.curriculum = curriculum
        self.k = k
        self.pop = []
        self.pop_size = pop_size
        self.xover_rate = xover_rate
        self.mutate_rate = mutate_rate
        self.best_so_far = None
        self.base_total_GPA = base_total_GPA
        self.base_total_credit = base_total_credit
        self.base_total_class = base_total_class

        #### Initialize variables ####
        self.pop = [Chromosome(self.chrm_len) for i in range(pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for chrm in self.pop:

            chrm.fitness, chrm.GPA, chrm.credit, chrm.curr = self.evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm


    def evolve(self):
        offspring = []

        #### Reproduction ####
        while len(offspring) < self.pop_size:
            #### Parent selection ####
            p1 = self.parent_selection(k=GA_K_TOURNAMENT)
            p2 = self.parent_selection(k=GA_K_TOURNAMENT)

            #### Crossover ####
            c1, c2 = Chromosome.crossover(p1, p2, self.xover_rate)

            #### Mutation ####
            Chromosome.mutate(c1, self.mutate_rate)
            Chromosome.mutate(c2, self.mutate_rate)

            offspring += [c1, c2]

        #### Evaluate offspring ####
        for i, chrm in enumerate(offspring):
            #print(f"Evaluating {(i / self.pop_size) * 100: 3.1f}%", end="\r")
            chrm.fitness, chrm.GPA, chrm.credit, chrm.curr = self.evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm

        #### Survival selection ####
        intermediate_pop = self.pop + offspring
        self.survival_selection(pool=intermediate_pop, n_survivor=self.pop_size)

    def parent_selection(self, k) -> Chromosome:

        champion = random.randint(0, (self.pop_size - 1))
        for i in range(k - 1):
            rand = random.randint(0, (self.pop_size - 1))
            if (self.pop[champion].fitness < self.pop[rand].fitness):
                champion = rand

        return self.pop[champion]
        # return self.pop[random.randint(0, (self.pop_size-1))]

    def survival_selection(self, pool, n_survivor) -> list:
        """
        I've implement this function for you. No need to change.
        """
        # If `pool` is the offspring only, this performs (mu, lambda) selection.
        # If `pool` is the union of the population and offspring, this performs (mu + lambda) selection.

        if n_survivor == len(pool):
            self.pop = pool
        else:
            sorted_pool = sorted(pool, key=lambda chrm: chrm.fitness, reverse=True)
            self.pop = sorted_pool[:n_survivor]
            
    def evaluate(self, gene: list) -> tuple:

        curr_copy = self.curriculum.copy()
        #curr_copy.iloc[:, :] = 0
        score = 0.0
        credit = 0
        total_GPA = 0
        class_num = 0
        GE_class_ub = 0
        GE_class_total = 0

        Matrix = [[0 for j in range(6)] for i in range (13)]

        for i in range(13):
            for j in range(5):
                if curr_copy.iloc[i, j] != 0:
                    #print(curr_copy.iloc[i, j])
                    Matrix[i][j] = 1
        '''
        for i in range(13):
            for j in range(6):
                if(self.curriculum.iloc[i, j] != 0):
                    Matrix[i][j] += 1
        '''
        ##     要求向度  1  2  3  4  自 社 人
        GE_condition = [0, 0, 0, 0, 0, 0, 0]
        for i in self.condition.general:
            if i != 'N':
                GE_class_ub += 1
            if i == '1' or i == '2' or i == '3' or i == '4':
                GE_condition[int(i)-1] += 1
            elif i == '自':
                GE_condition[4] += 1
            elif i == '社':
                GE_condition[5] += 1
            elif i == '人':
                GE_condition[6] += 1
        for i, index in enumerate(gene):
            if index == 1:
                class_num += 1
                if i < len(self.major):
                    credit += self.major.iloc[i, 2]
                    total_GPA += self.major.iloc[i, -3]
                    class_period = self.major.iloc[i, 3]
                else:
                    GE_class_total += 1
                    credit += self.GE.iloc[i-len(self.major), 3]
                    total_GPA += self.GE.iloc[i-len(self.major), -1]
                    class_period = self.GE.iloc[i-len(self.major), 4]
                    #print(self.GE.iloc[i-len(self.major), 2])
                    domain = self.GE.iloc[i-len(self.major), 2]
                    if domain == "Science":
                        GE_condition[4] -= 1
                    elif domain == "Society":
                        GE_condition[5] -= 1
                    elif domain == "Humanity":
                        GE_condition[6] -= 1
                    else:
                        GE_condition[int(domain)-1] -= 1
                for j, char in enumerate(str(class_period)):
                    if char in ['M', 'T', 'W', 'R', 'F']:
                        if char == 'M':
                            day_index = 0
                        elif char == 'T':
                            day_index = 1
                        elif char == 'W':
                            day_index = 2
                        elif char == 'R':
                            day_index = 3
                        elif char == 'F':
                            day_index = 4

                        if i < len(self.major):
                            time_index = self.major.iloc[i, 3][j + 1]
                        else:
                            time_index = self.GE.iloc[i-len(self.major), 4][j + 1]
                        if time_index == 'n':
                            time_index = 5
                        elif time_index == 'a':
                            time_index = 11
                        elif time_index == 'b':
                            time_index = 12
                        elif time_index == 'c':
                            time_index = 13
                        elif int(time_index) > 4:
                            time_index = int(time_index) + 1
                        else:
                            time_index = int(time_index)

                        Matrix[time_index-1][day_index] += 1
                        if i < len(self.major):
                            curr_copy.iloc[time_index - 1, day_index] = self.major.iloc[i, 1]
                        else:
                            curr_copy.iloc[time_index - 1, day_index] = self.GE.iloc[i-len(self.major), 1]
        if class_num != 0:
            averange_GPA = (total_GPA + self.base_total_GPA) / (class_num + self.base_total_class)
        else: averange_GPA = 0

        ## 每低1分GPA就扣100分，每高1分GPA就加100分
        if(averange_GPA < self.condition.GPA):
            score -= (self.condition.GPA - averange_GPA) * 500
        else:
            score += (averange_GPA - self.condition.GPA) * 500

        ## 每衝堂一節課就扣500分
        for i in Matrix:
            for j in i:
                if j > 1:
                    score -= 10000*(j-1)

        credit += self.base_total_credit
        ## 每多一或少一學分扣10分
        if credit > self.condition.credit_upperbound:
            score -= (credit - self.condition.credit_upperbound) * 200
        elif credit < self.condition.credit_lowerbound:
            score -= (self.condition.credit_lowerbound - credit) * 200

        ##每差一向度，扣50分
        for i in GE_condition:
            if i > 0:
                score -= i * 200

        #print("GE class total = " + str(GE_class_total))
        #print("GE class ub = " + str(GE_class_ub))
        ##如果通識課總數超過要求之通識課總數，扣500分
        if GE_class_total > GE_class_ub:
            score -= (GE_class_total - GE_class_ub) * 1000

        return score, averange_GPA, credit, curr_copy



#%%
#main
#load lesson dataset
PME_dataset = pd.read_csv(r'PME_course.csv')
ESS_dataset = pd.read_csv(r'ESS_course.csv')
GE_dataset = pd.read_csv(r'GE_course.csv')

#get the grade
PME_dataset['Grade'] = None
ESS_dataset['Grade'] = None
for i in range(len(PME_dataset)):
    PME_dataset.loc[i,'Grade'] = PME_dataset.loc[i,'Course No.'][-6]
for i in range(len(ESS_dataset)):
    ESS_dataset.loc[i,'Grade'] = ESS_dataset.loc[i,'Course No.'][-6]
    
#combine all lessons
dataset = pd.concat([PME_dataset,ESS_dataset,GE_dataset],join='outer').reset_index(drop=True)
#user input
my_condition = interface(list(dataset['Course No.']))
curriculum_pre, total_class, total_GPA, total_credit = lesson_required(my_condition.require,dataset)

if my_condition.general[0] != 'N':
    general_unique = np.unique(my_condition.general)
    general_unique = list(general_unique)
    for i in range(len(general_unique)):
        if general_unique[i] == '自':
            general_unique[i] = 'Science'
        if general_unique[i] == '社':
            general_unique[i] = 'Society'
        if general_unique[i] == '人':
            general_unique[i] = 'Humanity'



    GE_selected = GE_dataset[GE_dataset['Domain']==general_unique[0]]
    for i in range(len(general_unique)-1):
        temp = GE_dataset[GE_dataset['Domain']==general_unique[i+1]]
        GE_selected = pd.concat([GE_selected,temp])
else:
    GE_selected = GE_dataset.copy()
    
if my_condition.department == 'ESS':
    
    ESS_selected = ESS_dataset[ESS_dataset['Grade'] == str(my_condition.grade)].reset_index(drop=True)
    curriculum_final = GA(
        condition=my_condition,
        major_selected=ESS_selected,
        GE_selected=GE_selected,
        curriculum=curriculum_pre,
        pop_size=GA_POPULATION_SIZE,
        k=GA_K_TOURNAMENT,
        xover_rate=GA_CROSSOVER_RATE,
        mutate_rate=GA_MUTATION_RATE,
        base_total_GPA=total_GPA,
        base_total_credit=total_credit,
        base_total_class=total_class,
    )
elif my_condition.department == 'PME':
    
    PME_selected = PME_dataset[PME_dataset['Grade'] == str(my_condition.grade)].reset_index(drop=True)
    curriculum_final = GA(
        condition=my_condition,
        major_selected=PME_selected,
        GE_selected=GE_selected,
        curriculum=curriculum_pre,
        pop_size=GA_POPULATION_SIZE,
        k=GA_K_TOURNAMENT,
        xover_rate=GA_CROSSOVER_RATE,
        mutate_rate=GA_MUTATION_RATE,
        base_total_GPA=total_GPA,
        base_total_credit=total_credit,
        base_total_class=total_class,
    )

#curriculum_pre.to_csv('curriculum_pre.csv',encoding='utf_8_sig',index=None)

print(f"Generation {0: 4d}, best fitness = {curriculum_final.best_so_far.fitness:4.2f}")
curriculum_final.evaluate(curriculum_final.best_so_far.gene)
curriculum_final.best_so_far.print()

#### Evolve until termination criterion is met. #####
for i in range(GA_GENERATION):
    curriculum_final.evolve()
    print("\n")
    print(f"Generation {(i + 1): 4d}, best fitness = {curriculum_final.best_so_far.fitness:4.2f}")
    curriculum_final.evaluate(curriculum_final.best_so_far.gene)
    curriculum_final.best_so_far.print()

print("\n\n"
      "**********************\n"
      "Final Results:\n")
curriculum_final.best_so_far.print()
print("\n**********************")

curriculum_final.best_so_far.curr. to_csv("curriculum.csv", index = True, header = True, encoding = 'big5')
