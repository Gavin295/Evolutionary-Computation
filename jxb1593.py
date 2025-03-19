#!/usr/bin/env python3
import sys
import time
import random


def parse_assignment(assignment: str) -> list:
    return [bit=='1' for bit in assignment]


def is_clause_satisfied(assignment: list, clause: list) -> int:
    for literal in clause:
        index = abs(literal)-1
        if literal>0 and assignment[index]:
            return 1
        elif literal<0 and not assignment[index]:
            return 1
    return 0


#  Exercise 1
def parse_arguments_ex1():
    args = sys.argv
    clause = ""
    assignment = ""
    if "-clause" in args:
        clause = args[args.index("-clause") + 1]
    if "-assignment" in args:
        assignment = args[args.index("-assignment") + 1]
    return clause, assignment


def parse_clause_ex1(clause: str) -> list:
    values = clause.split()
    literals = []
    for val in values[1:-1]:  # ignore the first and last element
        try:
            literals.append(int(val))
        except ValueError:
            continue
    return literals


def validate_input_ex1(assignment: list, clause: list):
    max_var_index = max(abs(literal) for literal in clause) if clause else 0
    if len(assignment) < max_var_index:
        print("Error: Number of variables does not match", file=sys.stderr)
        sys.exit(1)


def exercise1():
    clause_str, assignment_str = parse_arguments_ex1()
    assignment = parse_assignment(assignment_str)
    clause = parse_clause_ex1(clause_str)
    validate_input_ex1(assignment, clause)
    result = is_clause_satisfied(assignment, clause)
    print(result)


#   Exercise 2
def parse_arguments_ex2():
    args = sys.argv
    wdimacs_file = ""
    assignment_str = ""

    if "-wdimacs" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
    if "-assignment" in args:
        assignment_str = args[args.index("-assignment") + 1]

    return wdimacs_file, assignment_str


def parse_assignment(assignment_str: str) -> list:
    """
    将形如 '01011' 的布尔赋值字符串转为 [False, True, False, True, True] 形式。
    - '1' -> True
    - '0' -> False
    """
    return [bit == '1' for bit in assignment_str]


def parse_wdimacs_file_ex2(filename: str):
    """
    解析 WDIMACS 或 WCNF 文件，读取变量数 n，返回子句列表。
    假设文件中有行:
      c ...            (注释行, 跳过)
      p wcnf n m ...   (格式行, 我们重点提取 n, 以及可选地 m)
      weight var1 var2 ... varK 0 (实际子句，首列weight，末尾0)
    """
    clauses = []
    num_variables = 0
    num_clauses = 0  # 可选看需不需要严格匹配

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # 跳过空行或注释行
            if not line or line.startswith('c'):
                continue

            # 如果是格式行，以 'p' 开头
            if line.startswith('p'):
                # 例如: p wcnf 4 2
                parts = line.split()
                # parts[0] = 'p'
                # parts[1] = 'wcnf' 或 'cnf'
                # parts[2] = n
                # parts[3] = m
                num_variables = int(parts[2])
                if len(parts) > 3:
                    num_clauses = int(parts[3])
                # 可以在此根据需要解析更多信息
                continue

            # 子句行: 例如 '1 2 -3 4 0' 或 '0 1 -2 3 0'
            tokens = list(map(int, line.split()))
            # 跳过首字段(通常是权重或0)，跳过末尾的 0
            clause_lits = tokens[1:-1]
            clauses.append(clause_lits)

    return num_variables, clauses


def validate_input_ex2(assignment: list, num_variables: int):
    """
    如果赋值长度和文件中声明的变量数不一致，则退出。
    """
    if len(assignment) != num_variables:
        print(0)
        sys.exit(1)


def is_clause_satisfied(assignment: list, clause: list) -> int:
    """
    判断单个子句是否被赋值 assignment 所满足。
    子句中若存在一个文字被满足，则子句被满足。
    返回 1 表示满足, 0 表示不满足。
    """
    for literal in clause:
        index = abs(literal) - 1
        # literal > 0  -> x_i
        # literal < 0  -> ¬x_i
        if literal > 0 and assignment[index]:
            # 该文字为正文字，且该位为 True，即满足子句
            return 1
        elif literal < 0 and not assignment[index]:
            # 该文字为负文字，且该位为 False，也满足子句
            return 1
    # 如果所有文字都不满足，则返回0
    return 0


def exercise2():
    wdimacs_file, assignment_str = parse_arguments_ex2()
    # 将字符串'0101...'转成bool列表
    assignment = parse_assignment(assignment_str)
    # 解析 WDIMACS 文件
    num_variables, clauses = parse_wdimacs_file_ex2(wdimacs_file)
    # 校验输入合法性
    validate_input_ex2(assignment, num_variables)
    # 计算满足的子句总数
    satisfied_count = sum(is_clause_satisfied(assignment, clause) for clause in clauses)
    # 输出结果
    print(satisfied_count)


#  Exercise 3
def parse_arguments_ex3():
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


def parse_wdimacs_file_ex3(filename: str):
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



def exercise3():
    wdimacs_file, time_budget, repetitions = parse_arguments_ex3()
    n, m, clauses = parse_wdimacs_file_ex3(wdimacs_file)
    results=[]

    for _ in range(repetitions):
        best_individual, best_nsat, used_time = run_memetic_bga(
            n, m, clauses,
            pop_size=50,
            mutation_rate=0.02,
            time_budget=time_budget,
            selection_rate=0.2,
            ls_iterations=30  # 局部搜索迭代次数
        )
        xbest_str = ''.join(str(bit) for bit in best_individual)
        results.append(f"{used_time}\t{best_nsat}\t{xbest_str}")

    for result in results:
        print(result)


def main():
    args=sys.argv
    if "-question" in args:
        try:
            question_index = args.index("-question")+1
            question_num = int(args[question_index])
        except (IndexError, ValueError):
            print("Invalid question parameter", file=sys.stderr)
            sys.exit(1)
        if question_num == 1:
            exercise1()
        elif question_num == 2:
            exercise2()
        elif question_num == 3:
            exercise3()
        else:
            print("error")
    else:
        print("No question specified", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
