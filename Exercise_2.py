import sys


def parse_arguments():
    args = sys.argv
    wdimacs_file = ""
    assignment = ""

    if "-wdimacs" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
    if "-assignment" in args:
        assignment = args[args.index("-assignment") + 1]

    return wdimacs_file, assignment


def parse_assignment(assignment: str) -> list:
    return [bit == '1' for bit in assignment]


def parse_wdimacs_file(filename: str):
    clauses = []
    num_variables = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('c') or line == '':
                continue
            if line.startswith('p'):

                parts = line.split()
                num_variables = int(parts[2])  # get the number of variables "n"
                continue
            else:
                literals = list(map(int, line.split()[:-1]))
                clauses.append(literals)

    return num_variables, clauses


def validate_input(assignment: list, num_variables: int):
    if len(assignment) != num_variables:
        print(0)
        sys.exit(1)


def is_clause_satisfied(assignment: list, clause: list) -> int:     # Check if the clause is satisfied
    for literal in clause:
        index = abs(literal) - 1
        if literal > 0 and assignment[index]:
            return 1
        elif literal < 0 and not assignment[index]:
            return 1
    return 0


if __name__ == "__main__":
    wdimacs_file, assignment_str = parse_arguments()
    assignment = parse_assignment(assignment_str)
    num_variables, clauses = parse_wdimacs_file(wdimacs_file)

    validate_input(assignment, num_variables)

    satisfied_count = sum(is_clause_satisfied(assignment, clause) for clause in clauses)
    print(satisfied_count)
