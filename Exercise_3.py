import sys
import time
import random

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
    n = 0  # number of variable
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
                # For normal clause lines, skip the first number (weight) and remove the trailing 0
                nums = list(map(int, line.split()))
                if len(nums) > 1:
                    clause_lits = nums[1:-1]
                    if len(clause_lits) > 0:
                        clauses.append(clause_lits)
    return n, m, clauses

def compute_nsat(individual, clauses):
    nsat = 0
    for clause in clauses:
        # If any literal in a clause is satisfied, the entire clause is satisfied.
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
        cost = m - nsat  # the smaller "cost" , the better
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
    generation = 0

    best_individual = None
    best_nsat_value = -1
    best_fitness_value = float('inf')

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

    time_spent_microseconds = int((time.time() - start_time) * 1_000_000)  # Convert elapsed time to microseconds
    return best_individual, best_nsat_value, time_spent_microseconds

if __name__ == "__main__":
    args = sys.argv
    if "-question" in args and int(args[args.index("-question") + 1]) == 3:
        wdimacs_file, time_budget, repetitions = parse_arguments()

        pop_size = 50
        mutation_rate = 1 / 50

        n, m, clauses = parse_wdimacs_file(wdimacs_file)

        results = []

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
    else:
        print("error")