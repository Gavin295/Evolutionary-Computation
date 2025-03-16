import sys


def parse_arguments():                                           # Parse command line parameters
    args = sys.argv
    clause = ""
    assignment = ""

    if "-clause" in args:
        clause = args[args.index("-clause") + 1]
    if "-assignment" in args:
        assignment = args[args.index("-assignment") + 1]

    return clause, assignment


def parse_assignment(assignment: str) -> list:
    return [bit == '1' for bit in assignment]


def parse_clause(clause: str) -> list:
    values = clause.split()
    literals = []
    for val in values[1:-1]:                                       # Ignore the first and last element
        try:
            literals.append(int(val))
        except ValueError:
            continue
    return literals


def validate_input(assignment: list, clause: list):                # Check if the number of variables matches
    max_var_index = max(abs(literal) for literal in clause) if clause else 0
    if len(assignment) < max_var_index:
        print(f"Error: Number of variables does not match", file=sys.stderr)
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
    clause_str, assignment_str = parse_arguments()
    assignment = parse_assignment(assignment_str)
    clause = parse_clause(clause_str)

    validate_input(assignment, clause)

    result = is_clause_satisfied(assignment, clause)
    print(result)
