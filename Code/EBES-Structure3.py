import copy
import random

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from Fitness_weight_original import calculate_fitness
import math

import Levy
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
# import matplotlib
# matplotlib.use('TkAgg')

class BES:
    def __init__(self, objective_function, bounds, solution_length, weight=0.3, swarm_size=50, max_iter=100, inertia_weight=0.4, num_grids=20,
                 max_len=100, draw=False):
        # print("+")
        self.objective_function = objective_function
        self.bounds = bounds
        self.weight = weight
        self.solution_length = solution_length
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.begin_weight = 0.9
        self.end_weight = 0.1
        self.dim = len(bounds)
        self.num_grids = num_grids
        self.max_len = max_len # pareto解集容量
        self.swarm = np.hstack([np.random.uniform(low, high, (swarm_size, 1)) for low, high in bounds])
        self.mins = -1
        self.maxs = 1
        # print("&")
        # print("Initial:",self.swarm,self.swarm.shape)
        self.velocity = np.zeros((swarm_size, self.dim))
        self.p_best = self.swarm.copy()
        self.hypercube_particle = self.swarm[0]
        # self.p_best_score = np.array([99999]*self.swarm_size)
        # print("^")
        self.swarm_score = np.array([self.objective_function(x,self.weight) for x in self.swarm])
        # print("%")
        self.iter_swarm_score = np.ones(self.swarm_size)
        self.partical_weight = np.ones(self.swarm_size)
        self.hypercube_particle_score = self.swarm_score[0]
        self.global_best_score = []
        self.global_best_count = []
        self.global_best_estimator_fitness = []
        self.draw = draw
        # print("*")

        # BES参数
        self.alpha = 1.5
        self.a = 10
        self.R = 1.5
        self.c1 = 1
        self.c2 = 1

    def tent(self,):
        x = np.zeros(self.solution_length)
        x[0] = random.random()
        for i in range(1, self.solution_length):
            if x[i - 1] <= 0.5:
                x[i] = 2 * x[i - 1]
            else:
                x[i] = 2 * (1 - x[i - 1])
        return x

    def Initial(self):
        # tent 映射
        for i in range(0, self.swarm_size):
            tent_array = self.tent()
            # temp_pop1 = [mins + (maxs - mins)*random.random() for j in range(individuallen)]
            temp_pop1 = [self.mins + (self.maxs - self.mins) * tent_array[j] for j in range(self.solution_length)]
            self.swarm[i] = temp_pop1

        self.p_best = self.swarm.copy()
        self.hypercube_particle = self.swarm[0]
        self.swarm_score = np.array([self.objective_function(x, self.weight) for x in self.swarm])
        self.hypercube_particle_score = self.swarm_score[0]

    # def Initial(self):
    #     # tent 映射
    #     pass


    def adaptive_weight(self,iter,score,global_best):
        omega = np.ones(self.swarm_size)
        k = (self.max_iter - iter)/self.max_iter
        delta_score = abs(score-global_best)
        for i in range(self.swarm_size):
            omega[i] = k*((self.begin_weight-self.end_weight)/(np.max(delta_score)))*delta_score[i]+self.end_weight
        return omega

    def swarm_mean(self):
        return self.swarm.mean(axis=0)

    def run(self):

        self.Initial()

        for iter in range(self.max_iter):
            print("----------------------------------Iter:",iter,"-----------------------------------------")

            # select
            p_mean = self.swarm_mean()
            levy = Levy.levy_flight(self.swarm_size, self.solution_length, 1.5)
            for i in range(self.swarm_size):
                swarm_new = self.hypercube_particle + self.alpha*random.random()*(p_mean-self.swarm[i])*levy[i]
                swarm_new = np.array(
                    [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])
                swarm_score = self.objective_function(swarm_new,self.weight)
                # print(self.swarm_score.shape)
                # if swarm_score[0] < self.swarm_score[i][0]:
                #     self.swarm[i] = swarm_new
                #     self.swarm_score[i] = swarm_score

                if swarm_score[0] < self.hypercube_particle_score[0]:
                    self.hypercube_particle = swarm_new
                    self.hypercube_particle_score = swarm_score
                    print("Change in select:", self.hypercube_particle_score)

            # search
            xi_array = np.ones(self.swarm_size)
            yi_array = np.ones(self.swarm_size)
            # w = 0.9 - 0.5 * pow(math.e, pow(iter / self.max_iter, 2))
            w = 3 + (1) * np.sin(math.pi + ((iter / self.max_iter) * (1 / 2) * math.pi))
            # w = math.sin(math.pi+i)
            for i in range(self.swarm_size):
                theta = self.a * math.pi * random.random()
                r = theta + self.R * random.random()
                xi_array[i] = r * math.sin(theta)
                yi_array[i] = r * math.cos(theta)

            p_mean = self.swarm_mean()
            for i in range(self.swarm_size):
                xi = xi_array[i]/np.max(xi_array)
                yi = yi_array[i]/np.max(yi_array)
                if i != (self.swarm_size - 1):
                    swarm_new = self.swarm[i] + w*xi*(self.swarm[i]-p_mean) + w*yi*(self.swarm[i]-self.swarm[i+1])
                else:
                    swarm_new = self.swarm[i] + w*xi * (self.swarm[i] - p_mean) + w*yi * (
                                self.swarm[i] - self.swarm[0])

                swarm_new = np.array(
                    [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])

                # swarm_new = np.array(
                #     [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])
                swarm_score = self.objective_function(swarm_new, self.weight)
                # if swarm_score[0] < self.swarm_score[i][0]:
                #     self.swarm[i] = swarm_new
                #     self.swarm_score[i] = swarm_score

                if swarm_score[0] < self.hypercube_particle_score[0]:
                    self.hypercube_particle = swarm_new
                    self.hypercube_particle_score = swarm_score
                    print("Change in search:", self.hypercube_particle_score)


            # swoop
            xi_array = np.ones(self.swarm_size)
            yi_array = np.ones(self.swarm_size)
            for i in range(self.swarm_size):
                theta = self.a * math.pi * random.random()
                r = theta
                xi_array[i] = r * math.sinh(theta)
                yi_array[i] = r * math.cosh(theta)

            p_mean = self.swarm_mean()
            for i in range(self.swarm_size):
                xi = xi_array[i]/np.max(xi_array)
                yi = yi_array[i]/np.max(yi_array)

                delta_x = xi*(self.swarm[i]-self.c1*p_mean)
                delta_y = yi*(self.swarm[i]-self.hypercube_particle)

                swarm_new = random.random()*self.hypercube_particle+delta_x+delta_y
                swarm_new = np.array(
                    [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])

                # swarm_new = np.array(
                #     [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])

                # levy = Levy.levy_flight(self.swarm_size, self.solution_length, 1.5)
                r = random.random()
                # if r < 0.5:
                #     swarm_new = self.hypercube_particle + 0.5*swarm_new
                #
                # else:
                #     swarm_new = self.hypercube_particle + 0.5*levy[i]*(self.hypercube_particle-levy[i]*swarm_new)

                if r < 0.2:
                    # swarm_new = self.hypercube_particle + 0.5*swarm_new
                    swarm_new = swarm_new
                elif 0.2 <r < 0.6:
                    swarm_new = self.hypercube_particle*(1+np.random.standard_normal(self.solution_length))
                else:
                    # swarm_new = self.hypercube_particle + 0.5*levy[i]*(self.hypercube_particle-levy[i]*swarm_new)
                    swarm_new = np.random.standard_normal(self.solution_length)*(self.hypercube_particle-swarm_new)+\
                                np.random.standard_normal(self.solution_length)*(self.swarm[random.randint(0,self.swarm_size-1)]-swarm_new)
                swarm_new = np.array(
                    [np.clip(x, bound[0], bound[1]) for x, bound in zip(swarm_new, self.bounds)])
                # print(swarm_new)
                swarm_score = self.objective_function(swarm_new, self.weight)
                # self.swarm[i] = swarm_new
                # self.swarm_score[i] = swarm_score
                # if swarm_score[0] < self.swarm_score[i][0]:
                #     self.swarm[i] = swarm_new
                #     self.swarm_score[i] = swarm_score

                if swarm_score[0] < self.hypercube_particle_score[0]:
                    self.hypercube_particle = swarm_new
                    self.hypercube_particle_score = swarm_score
                    print("Change in swoop:", self.hypercube_particle_score)

            # back
            t_hypercube_particle = copy.deepcopy(self.hypercube_particle)
            swarm_new = np.array([1]*self.solution_length) + np.array([-1]*self.solution_length) - t_hypercube_particle
            swarm_score = self.objective_function(swarm_new,self.weight)
            if swarm_score[0] < self.hypercube_particle_score[0]:
                self.hypercube_particle = swarm_new
                self.hypercube_particle_score = swarm_score
                print("Change in back:", self.hypercube_particle_score)

            self.global_best_score.append(self.hypercube_particle_score[0])
            self.global_best_count.append(self.hypercube_particle_score[1])
            self.global_best_estimator_fitness.append(self.hypercube_particle_score[2])



            print("Best Fitness:",self.hypercube_particle_score[0], "Best Count:", self.hypercube_particle_score[1],
                  "Best estimator fitness:", self.hypercube_particle_score[2])
            # print(self.partical_weight)
            # print(hypercube_partical)
            # print(self.hypercube_particle)

        print("Global Fitness:",self.global_best_score)

        return self.hypercube_particle_score[0], self.hypercube_particle_score[1], self.hypercube_particle_score[2], self.hypercube_particle



if __name__ == "__main__":
    #读取数据
    data1 = pd.read_csv('../CSVDataset/automobile.csv')
    # data1 = pd.read_excel('D:\\论文\\论文数据\\实验处理得到的数据\\ASD\\特征选择个数\\50_0_117833_new.xlsx')
    columns_names = data1.columns[:-1].values
    print("columns_names:", columns_names, len(columns_names))
    data = np.array(data1.iloc[:, :])
    data = pd.DataFrame(data)  # 从第一行第一列开始
    X = data.iloc[:, :-1]  # 数据部分
    print("-----------------------------数据值-----------------------------------", )
    print(data.iloc[:, :-1])  # 目标值
    print("----------------------------+目标值+------------------------------------")
    y = data.iloc[:, -1:].values.flatten()
    print(y)
    chrom_length = len(data.columns) - 1



    ZDT1_bounds = [(-5, 5) for i in range(chrom_length)]
    # mopso = BES(calculate_fitness, ZDT1_bounds,solution_length=chrom_length, weight=0.2, draw=True, swarm_size=25, max_iter=200, num_grids=10, max_len=20)
    # print("-")
    # mopso.run()

    repeat = 30
    repeat_fitness = np.ones(repeat)
    repeat_count = np.ones(repeat)
    repeat_estimator_fitness = np.ones(repeat)

    for i in range(repeat):
        mopso = BES(calculate_fitness, ZDT1_bounds,solution_length=chrom_length, weight=0.2, draw=True, swarm_size=25, max_iter=200, num_grids=10, max_len=20)
        print("------------------------------------------------Repeat:", i,
              "-----------------------------------------------")
        fitness, count, estimator_fitness, particle = mopso.run()
        repeat_fitness[i] = fitness
        repeat_count[i] = count
        repeat_estimator_fitness[i] = estimator_fitness
        print("Repeat fitness:", repeat_fitness)
        print("Repeat count:", repeat_count)
        print("repeat estimator fitness:", repeat_estimator_fitness)
        print("Particle:", particle)

    print("Best Fitness:", np.min(repeat_fitness), "Worst Fitness:", np.max(repeat_fitness),
          "Mean Fitness:", np.mean(repeat_fitness), "SD:", np.std(repeat_fitness))

    print("Best Count:", np.min(repeat_count), "Worst Count:", np.max(repeat_count),
          "Mean Count:", np.mean(repeat_count), "SD:", np.std(repeat_count))

    print("Best Estimator Fitness:", np.min(repeat_estimator_fitness), "Worst Estimator Fitness:",
          np.max(repeat_estimator_fitness),
          "Mean Estimator Fitness:", np.mean(repeat_estimator_fitness), "SD:", np.std(repeat_estimator_fitness))

    # print("Iter Score:",mopso.global_best_score)
    # print("Iter Count:",mopso.global_best_count[-1])
    # print("Iter estimator fitness:",mopso.global_best_estimator_fitness[-1])

    # import numpy as np
    # import pandas as pd
    #
    # data = pd.DataFrame(mopso.global_best_score)
    #
    # writer = pd.ExcelWriter('PSO.xlsx')  # 写入Excel文件
    # data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # writer.save()
    #
    # writer.close()
    #
    iter = np.arange(1,201,1)

    plt.plot(iter,mopso.global_best_score,marker='o',color = 'blue',label='PSO')

    plt.show()

