import numpy as np
from deap import algorithms, base, creator, tools

# 定义目标函数(ZDT1)
def zdt1(individual):
    f1 = individual[0]
    g = 1.0 + 9.0 * np.sum(individual[1:]) / (len(individual) - 1)
    f2 = g * (1.0 - (f1 / g) ** 0.5)
    return f1, f2

# 创建个体类和种群类
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 注册工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0/30)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", zdt1)

# 创建初始种群
pop = toolbox.population(n=50)

# 配置算法参数
CXPB, MUTPB, NGEN = 0.6, 0.3, 50  # 调整交叉和变异概率

# 统计设置
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# 运行NSGA-II算法
algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, verbose=True)

import matplotlib.pyplot as plt

# 从算法结果中提取Pareto最优解
pareto_front = tools.sortNondominated(pop, len(pop))[-1]

# 提取每个解的目标函数值
fitnesses = [ind.fitness.values for ind in pareto_front]

# 分解为两个目标函数的值
fitnesses = np.array(fitnesses)
objs = fitnesses.T

# 绘制Pareto前沿
plt.figure(figsize=(10, 6))
plt.scatter(objs[0], objs[1], c='green', alpha=0.7)
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front')
plt.grid(True)
plt.show()

'''
import numpy as np
from deap import algorithms, base, creator, tools

# 定义目标函数
def zdt1(individual):
    f1 = individual[0]
    g = 1.0 + 9.0 * np.sum(individual[1:]) / (len(individual) - 1)
    f2 = g * (1.0 - (f1 / g) ** 0.5)
    return f1, f2

# 创建个体类和种群类
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 注册工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=0.0, up=1.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=0.0, up=1.0, indpb=1.0/30)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", zdt1)

# 创建初始种群
pop = toolbox.population(n=50)

# 配置算法参数
CXPB, MUTPB, NGEN = 0.9, 1.0, 50

# 统计设置
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# 运行NSGA-II算法
algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=50, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, verbose=True)
'''

'''
error reason
代码中的交叉CXPB和变异MUTPB概率之和超过了1.这导致了AssertionError。在遗传算法中.每个个体在一个世代中有两种可能的变化：交叉或变异。因此.交叉和变异的概率之和应该小于或等于1。
我将修正这个问题.并将CXPB和MUTPB的值调整为合法的范围内。以下是修正后的代码：
'''