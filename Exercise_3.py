import sys
import time
import random
import matplotlib.pyplot as plt


def parse_arguments():
    args = sys.argv
    wdimacs_file = ""
    time_budget = 0
    repetitions = 1

    if "-wdimacs" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
    if "-time_budget" in args:
        time_budget = int(args[args.index("-time_budget") + 1])
    if "-repetitions" in args:
        repetitions = int(args[args.index("-repetitions") + 1])
    return wdimacs_file, time_budget, repetitions


def parse_wdimacs_file(filename: str):
    clauses = []
    n = 0  # number of variables
    m = 0  # number of clauses
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # 过滤空行或注释
            if line.startswith('c') or line == "":
                continue
            if line.startswith('p'):
                parts = line.split()
                n = int(parts[2])
                m = int(parts[3])
            else:
                # 对于普通子句行，跳过第一个数字（可能是权重）并去掉末尾的 0
                nums = list(map(int, line.split()))
                if len(nums) > 1:
                    clause_lits = nums[1:-1]  # 注意：nums[0]通常是子句权重，可以忽略
                    if clause_lits:
                        clauses.append(clause_lits)
    return n, m, clauses


def compute_nsat(individual, clauses):
    """计算给定个体(individual)满足的子句数。"""
    nsat = 0
    for clause in clauses:
        # 如果子句中任意literal被满足，则整个子句满足
        for literal in clause:
            var_idx = abs(literal) - 1
            sign = (literal > 0)
            # individual[var_idx] == 1 => 变量取真，==0 => 变量取假
            if (sign and individual[var_idx] == 1) or (not sign and individual[var_idx] == 0):
                nsat += 1
                break
    return nsat


def evaluate_fitness(population, clauses, m):
    """评估整个种群的适应度(此处以cost = m - nsat)，并返回fitness_scores与其对应的nsat值列表。"""
    fitness_scores = []
    nsat_list = []
    for individual in population:
        nsat = compute_nsat(individual, clauses)
        cost = m - nsat  # cost越小越好；也可直接把nsat作为fitness(越大越好)
        fitness_scores.append(cost)
        nsat_list.append(nsat)
    return fitness_scores, nsat_list


def initialize_population(pop_size, n):
    """随机初始化种群。"""
    return [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]


def select_parents(population, fitness_scores, selection_rate):
    """简单的截断选择：根据fitness从好到差排序，选出一定比例的个体作为父代。"""
    ranked_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0])  # cost越小越好
    num_parents = max(int(len(population) * selection_rate), 2)
    return [ind for (_, ind) in ranked_pairs[:num_parents]]


def crossover(parents):
    """随机从父母集中选两条染色体做单点交叉，产生后代。"""
    offspring = []
    for _ in range(len(parents)):
        if len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
            cp = random.randint(1, len(p1) - 1)
            child1 = p1[:cp] + p2[cp:]
            child2 = p2[:cp] + p1[cp:]
            offspring.append(child1)
            offspring.append(child2)
    return offspring


def mutate(offspring, mutation_rate):
    """对后代进行变异。"""
    for individual in offspring:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
    return offspring


def local_search(individual, clauses, m, max_iterations=30):
    """
    采用随机单变量翻转的启发式局部搜索:
    - 每次随机选择一个变量翻转
    - 如果满足子句数提高则接受，否则回退
    - 重复若干次
    """
    current = individual[:]
    best_nsat = compute_nsat(current, clauses)
    for _ in range(max_iterations):
        idx = random.randrange(len(current))
        # 尝试翻转
        current[idx] = 1 - current[idx]
        new_nsat = compute_nsat(current, clauses)
        if new_nsat > best_nsat:
            best_nsat = new_nsat
        else:
            # 回退翻转
            current[idx] = 1 - current[idx]
    return current


def find_best_solution(population, fitness_scores, nsat_list):
    best_idx = min(range(len(population)), key=lambda i: fitness_scores[i])
    return population[best_idx], nsat_list[best_idx], fitness_scores[best_idx]


def run_memetic_bga(n, m, clauses, pop_size, mutation_rate, time_budget, selection_rate, ls_iterations=30):
    """
    在原有BGA(遗传算法)的基础上，每轮在产生后代后，使用local_search对其进行强化，
    最后在有限time_budget内找出最优解。
    """
    start_time = time.time()
    population = initialize_population(pop_size, n)
    generation = 0

    best_individual = None
    best_nsat_value = -1
    best_fitness_value = float('inf')

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break

        # 评估当前种群
        fitness_scores, nsat_list = evaluate_fitness(population, clauses, m)
        candidate_best, candidate_nsat, candidate_fit = find_best_solution(population, fitness_scores, nsat_list)

        # 更新全局最优解
        if candidate_fit < best_fitness_value:
            best_fitness_value = candidate_fit
            best_nsat_value = candidate_nsat
            best_individual = candidate_best

        # 选择父代
        parents = select_parents(population, fitness_scores, selection_rate)
        # 交叉产生后代
        offspring = crossover(parents)
        # 变异
        offspring = mutate(offspring, mutation_rate)
        # 局部搜索 (Memetic 核心)
        for i in range(len(offspring)):
            offspring[i] = local_search(offspring[i], clauses, m, ls_iterations)

        # 新种群 = 父代 + 后代(截断到pop_size大小)
        population = (parents + offspring)[:pop_size]
        generation += 1

    time_spent_microseconds = int((time.time() - start_time) * 1_000_000)
    return best_individual, best_nsat_value, time_spent_microseconds


if __name__ == "__main__":
    args = sys.argv
    # 如果使用 -question 3，就执行Memetic BGA代码
    if "-question" in args and int(args[args.index("-question") + 1]) == 3:
        wdimacs_file, time_budget, repetitions = parse_arguments()
        n, m, clauses = parse_wdimacs_file(wdimacs_file)

        # 参数可以根据需要调节
        pop_size = 50
        mutation_rate = 0.02
        selection_rates = [0.2]
        all_nsat_results = []

        for selection_rate in selection_rates:
            nsat_results = []
            for _ in range(repetitions):
                best_individual, best_nsat, used_time = run_memetic_bga(
                    n, m, clauses,
                    pop_size=pop_size,
                    mutation_rate=mutation_rate,
                    time_budget=time_budget,
                    selection_rate=selection_rate,
                    ls_iterations=30   # 局部搜索迭代次数
                )
                nsat_results.append(best_nsat)

                xbest_str = "".join(str(bit) for bit in best_individual)
                print(f"{used_time} {best_nsat} {xbest_str}")
            all_nsat_results.append(nsat_results)

        # 箱线图绘制
        plt.figure()
        plt.boxplot(all_nsat_results, labels=[str(int(sr*100))+"%" for sr in selection_rates])
        plt.xlabel('Parent Selection Rate')
        plt.ylabel('Number of Satisfied Clauses')
        plt.title(f'Boxplot of Satisfied Clauses over 100 runs')
        plt.show()
    else:
        print("error")
