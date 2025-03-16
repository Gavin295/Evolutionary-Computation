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
    assignment = ""
    if "-wdimacs" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
    if "-assignment" in args:
        assignment = args[args.index("-assignment") + 1]
    return wdimacs_file, assignment


def parse_wdimacs_file_ex2(filename: str):
    clauses = []
    num_variables = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('c') or line == '':
                continue
            if line.startswith('p'):
                parts = line.split()
                num_variables = int(parts[2])  # get the number of variables: "n"
                continue
            else:
                literals = list(map(int, line.split()[:-1]))
                clauses.append(literals)
    return num_variables, clauses


def validate_input_ex2(assignment: list, num_variables: int):
    if len(assignment) != num_variables:
        print(0)
        sys.exit(1)


def exercise2():
    wdimacs_file, assignment_str = parse_arguments_ex2()
    assignment = parse_assignment(assignment_str)
    num_variables, clauses = parse_wdimacs_file_ex2(wdimacs_file)
    validate_input_ex2(assignment, num_variables)
    satisfied_count = sum(is_clause_satisfied(assignment, clause) for clause in clauses)
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
    m = 0  # number of clause
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('c') or line == "":
                continue
            if line.startswith('p'):
                parts = line.split()
                n = int(parts[2])
                m = int(parts[3])
            else:
                nums = list(map(int, line.split()))
                if len(nums) > 1:
                    clause_lits = nums[1:-1]
                    if clause_lits:
                        clauses.append(clause_lits)
    return n, m, clauses

def compute_nsat(individual, clauses):
    nsat = 0
    for clause in clauses:
        for literal in clause:
            var_idx = abs(literal) - 1
            sign = (literal > 0)
            if var_idx >= len(individual):
                continue
            if (sign and individual[var_idx] == 1) or (not sign and individual[var_idx] == 0):
                nsat += 1
                break
    return nsat


def evaluate_fitness(population, clauses, m):
    fitness_scores = []
    nsat_list = []
    for individual in population:
        nsat = compute_nsat(individual, clauses)
        cost = m - nsat
        fitness_scores.append(cost)
        nsat_list.append(nsat)
    return fitness_scores, nsat_list


def initialize_population(pop_size, n):
    return [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]


def select_parents(population, fitness_scores):
    ranked_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0])
    num_parents = max(int(len(population) * 0.3), 2)
    return [ind for (_, ind) in ranked_pairs[:num_parents]]


def crossover(parents):
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
    for individual in offspring:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
    return offspring


def find_best_solution(population, fitness_scores, nsat_list):
    best_idx = min(range(len(population)), key=lambda i: fitness_scores[i])
    return population[best_idx], nsat_list[best_idx], fitness_scores[best_idx]


def run_standard_bga(n, m, clauses, pop_size, mutation_rate, time_budget):
    start_time = time.time()
    population = initialize_population(pop_size, n)

    # 初始化时直接使用种群中第一个个体作为最优解
    best_individual = population[0]
    best_nsat_value = compute_nsat(population[0], clauses)
    best_fitness_value = m - best_nsat_value

    generation = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break

        fitness_scores, nsat_list = evaluate_fitness(population, clauses, m)
        candidate_best, candidate_nsat, candidate_fit = find_best_solution(population, fitness_scores, nsat_list)

        if candidate_fit < best_fitness_value:
            best_fitness_value = candidate_fit
            best_nsat_value = candidate_nsat
            best_individual = candidate_best

        parents = select_parents(population, fitness_scores)
        offspring = crossover(parents)
        offspring = mutate(offspring, mutation_rate)
        population = (parents + offspring)[:pop_size]

        generation += 1

    time_spent_microseconds = int((time.time() - start_time) * 1_000_000)  # 转换运行时间为微秒
    return best_individual, best_nsat_value, time_spent_microseconds


def exercise3():
    wdimacs_file, time_budget, repetitions = parse_arguments_ex3()
    pop_size = 50
    mutation_rate = 1 / 50

    n, m, clauses = parse_wdimacs_file_ex3(wdimacs_file)

    results=[]

    for _ in range(repetitions):
        best_individual, best_nsat, used_time = run_standard_bga(
            n, m, clauses,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            time_budget=time_budget
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