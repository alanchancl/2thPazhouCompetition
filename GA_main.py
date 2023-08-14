## 读取数据
import pandas as pd
import numpy as np

# df_raw = pd.read_excel(
#     '数据集A榜基于多目标、多源数据扰动预测的智能排序算法.xlsx',
#     sheet_name='处理后数据',
#     header=1,
#     engine='openpyxl')

df_raw = pd.read_csv('answer_77.59533.csv')
# df_raw[['计划日期', '车型', '天窗', '外色描述', '车辆等级描述', '电池特征']]

## 格式化输入数据
COLUMNS = {'date': '计划日期', 'kind': '车型', 'roof': '天窗', 'color': '外色描述', 'power': '电池特征', 'grade': '车辆等级描述'}
VALUES = {}
columns = []

for c, name in COLUMNS.items():
    values = list(df_raw[name].unique())
    VALUES[c] = values

    s = df_raw[name].map({v: i for i, v in enumerate(values)})
    s.name = c
    columns.append(s)

# 大颜色
MAJOR = frozenset(
    i for i, c in enumerate(VALUES['color'])
    if c in ["白云蓝","极地白","极地白-Y","幻影银","幻影银(出租车)","极速银","极速银-Y","极速银(出租车)","夜影黑","夜影黑-Y","自由灰","自由灰-Y","素雅灰","素雅灰-Y","天青色","天青色-Y","珍珠白","全息银"])
# 小颜色
MINOR = frozenset(
    i for i, c in enumerate(VALUES['color'])
    if c in ["量子红","量子红-Y","冰玫粉","冰玫粉-Y","蒂芙尼蓝","星漫绿","星漫绿-Y","琉璃红","夜荧黄","黄绿荧","薄荷贝绿","烟雨青","幻光紫","广交红","闪电橙","脉冲蓝","天际灰","火焰橙","幻光紫","幻光紫-Y","琉璃红","松花黄","松花黄-Y"])
# 双色
BICOLOR = frozenset(
    i for i, c in enumerate(VALUES['color'])
    if '/' in c)
# 石墨
GRAPHITE = frozenset(
    i for i, c in enumerate(VALUES['power'])
    if '石墨' in c)
# 四驱
AWD = frozenset(
    i for i, c in enumerate(VALUES['power'])
    if '/' in c)
# K3
K3 = frozenset(
    i for i, c in enumerate(VALUES['kind'])
    if c in ['K3'])

df_converted = pd.concat(columns,axis=1)
df_converted['awd'] = df_converted['power'].isin(AWD)
df_converted['major'] = np.where(df_converted['color'].isin(MAJOR), df_converted['color'], -1)
df_converted['minor'] = np.where(df_converted['color'].isin(MINOR), df_converted['color'], -1)
df_converted['bicolor'] = np.where(df_converted['color'].isin(BICOLOR), df_converted['color'], -1)
df_converted['graphite'] = np.where(df_converted['power'].isin(GRAPHITE), df_converted['power'], -1)
df_converted['k3'] = np.where(df_converted['kind'].isin(K3), df_converted['kind'], -1)

## 评分

# 切换次数

# def eval_score(indices):
#     df_all = df_converted.take(indices)
#     df = pd.DataFrame(np.vstack([_eval_score(df_all[df_all['date'] == i]) for i in range(53)]))
#     df.columns = [i+1 for i in range(17)]
#     return df

## 遗传算法优化
from deap import base, creator, tools, algorithms
import random

class GA_Solution:
    population_size = 5000
    num_generations = 5000

    def __init__(self, i):
        self.date = VALUES['date'][i]
        self.df = df_converted[df_converted['date'] == i].reset_index(drop=True)
        self.num_rows = None
        # if (VALUES['date'][i]==2022-10-28):
        #     print(VALUES['date'][i])
        df_group = df_raw[df_raw['计划日期'] == self.date]
        df_group = df_group[['车型', '天窗', '外色描述', '车辆等级描述', '电池特征']]
        df_group = df_group.reset_index(drop=True)
        self.grouped_indices_list = df_group.groupby(df_group.columns.tolist()).apply(lambda x:  x.index.tolist()).tolist()
        self.num_rows = len(self.grouped_indices_list)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.sample, range(self.num_rows), self.num_rows)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxPartialyMatched)  # 部分匹配交叉
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # 随机交换部分城市
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.log = []

    def evaluate_fitness(self, individual):
        new_grouped_indices_list = [self.grouped_indices_list[i] for i in individual]
        # 将二维数组变成一维数组
        new_grouped_indices_list = np.concatenate(new_grouped_indices_list).tolist()
        weight_coefficient = np.array([50, 4, 2, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # weight_coefficient = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        df_reordered = self.df.iloc[new_grouped_indices_list]
        fitness = np.sum(self._eval_score(df_reordered) * weight_coefficient.flatten())
        return fitness,

    def switch_count(self, a, b=None):
        d = np.diff(a)
        if b is not None:
            d = np.where(np.diff(b)==0, d, 0)
        return np.count_nonzero(d)

    def _eval_score(self, df):
        df = df[['kind','roof','color','power','grade','awd','major','minor','bicolor','graphite','k3']].to_numpy()
        kind,roof,color,power,grade,awd,major,minor,bicolor,graphite,k3 = [df[:,i] for i in range(11)]
        s = [
            self.switch_count(c1, c2)
            for c1, c2 in [
                (kind,None),  # 车型 - 切换
                (roof,None),  # 天窗 - 切换
                (color,None), # 颜色 - 切换
                (power,None), # 电池 - 切换
                (grade,None), # 配置等级 - 切换
                (awd, kind),  # 四驱 - 单批内集中
            ]
        ] + [
            0 # K3 - 单批内集中 (不知所云)
        ]

        gaps = []
        counts = []
        evens = []

        for c, glb, clb, cub, e in [
            (minor,    60,   15,   30,   False),   # 小颜色 - 批与批间隔, 单批内数量
            (bicolor,  60,   None, 4,    True),    # 双色车 - 批与批间隔, 单批内数量, 均匀性
            (major,    None, 15,   None, False),   # 大颜色 - 单批内数量
            (graphite, 30,   None, 1,    True),    # 石墨电池 - 批与批间隔, 单批内数量, 均匀性
            (k3,       None, None, None, True),    # K3 - 均匀性
            ]:
            # 批的起始位置
            p, = np.where(np.diff(c, prepend=-1,append=-1))
            # 批是否为小颜色，除掉开头末尾非小颜色的
            b = c[p[:-1]] >= 0
            # 小颜色批数
            total = np.count_nonzero(b)

            have_gap = glb is not None
            if total < 2:
                if have_gap:
                    gaps.append(0)
                if e:
                    gv = 0.0
            elif e or have_gap:
                # 小颜色批的间隔 (小颜色批的起始位置 - 上一个小颜色批的结束位置)
                g = np.extract(b, p)[1:] - np.extract(b, p[1:])[:-1]
                if have_gap:
                    gaps.append(np.count_nonzero(g < glb)/len(g))
                if e:
                    # gv = np.std(g)/np.mean(g)
                    gv = np.nanstd(g) / np.nanmean(g)

            have_count = (clb is not None) or (cub is not None)
            if total < 1:
                if have_count:
                    counts.append(0)
                nv = 0.0
            elif e or have_count:
                # 小颜色各批的数量
                n = np.extract(b, np.diff(p))
                if have_count:
                    count = (
                        (0 if clb is None else np.count_nonzero(n < clb)) +
                        (0 if cub is None else np.count_nonzero(n > cub)))
                    counts.append(count / total)
                if e:
                    # nv = np.std(n)/np.mean(n)
                    nv = np.nanstd(n) / np.nanmean(n)

            if e:
                evens.append(nv+gv)

        s.extend(gaps)
        s.extend(counts)
        s.extend(evens)
        return np.array(s)

    def GA_find_better_answer(self):
        if self.num_rows < 2:
            best_individual = [0]
            Orig_individual = [0]
        else:

            population = self.toolbox.population(n=self.population_size)
            Orig_individual = creator.Individual(np.arange(self.num_rows).copy())
            for i in range(self.population_size//2):
                population[i] = Orig_individual

            stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
            stats.register('avg', np.mean)
            stats.register('min', np.min)
            stats.register('max', np.max)

            final_population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.num_generations, stats=stats, halloffame=None, verbose=False)

            best_individual = tools.selBest(final_population, k=1)[0]

            # # 获取每一代的统计信息
            # gen = logbook.select('gen')
            # fit_avg = logbook.select('avg')
            # fit_min = logbook.select('min')
            # fit_max = logbook.select('max')

            # # 显示每一代的最佳适应度值和其他统计信息
            # for g, avg, min_, max_, best_fit in zip(gen, fit_avg, fit_min, fit_max, best_fitness):
            #     print(f"Generation {g}: Average Fitness: {avg}, Min Fitness: {min_}, Max Fitness: {max_}, Best Fitness: {best_fitness}")


        best_path = np.concatenate([self.grouped_indices_list[i] for i in best_individual]).tolist()

        # print("Date", self.date)
        # print("Orig distance:", self.evaluate_fitness(Orig_individual)[0])
        # print("Best distance:", self.evaluate_fitness(best_individual)[0])
        self.log.append([self.date,self.evaluate_fitness(Orig_individual)[0],self.evaluate_fitness(best_individual)[0]])
        return best_path,self.evaluate_fitness(best_individual)[0]
    
    def Get_log(self):
        return self.log[0]
    
result = []
Loss = []
log = []
for i in range(53):
    ga_solution = GA_Solution(i)
    best_result, best_loss = ga_solution.GA_find_better_answer()
    log.append(ga_solution.Get_log())
    result.append(best_result)
    Loss.append(best_loss)

result = np.concatenate([df_converted.index[df_converted['date']==i].take(index) for i, index in enumerate(result)])
print(sum(Loss))
print(log)

## 输出结果
from io import StringIO

buf = StringIO()
df_raw.take(result).to_csv(buf,index=False)

from zipfile import ZipFile
with ZipFile('data/answer.zip', 'w') as myzip:
        with myzip.open('answer.csv', 'w') as myfile:
            myfile.write(buf.getvalue().encode())