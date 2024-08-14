import concurrent.futures
import random

import pulp
import click
import numpy as np
import pandas as pd


def generate_binary_matrix(X, Y, seed):
    # Number of rows and columns
    nrows = len(X)
    ncols = len(Y)

    # Create a LP problem with PuLP
    prob = pulp.LpProblem("BinaryMatrix", pulp.LpMaximize)  # We're not actually maximizing anything specific

    # Create a dictionary of pulp variables with keys from 0 to nrows*ncols
    # Each variable is binary, indicating whether the cell is 1 (True) or 0 (False)
    choices = pulp.LpVariable.dicts("Choice", (range(nrows), range(ncols)), cat='Binary')

    # Row sum constraints
    for i in range(nrows):
        prob += pulp.lpSum([choices[i][j] for j in range(ncols)]) == X[i], f"RowSum_{i}"

    # Column sum constraints
    for j in range(ncols):
        prob += pulp.lpSum([choices[i][j] for i in range(nrows)]) == Y[j], f"ColumnSum_{j}"

    # The problem is solved using PuLP's choice of solver
    prob.solve(pulp.PULP_CBC_CMD(msg=False, options= [f"RandomS {seed}", f"RandomC {seed}"]))

    # The status of the solution is printed to the screen
    print("Status:", pulp.LpStatus[prob.status])

    # If the problem has a feasible solution, construct the matrix
    if pulp.LpStatus[prob.status] == 'Optimal':
        # Create an empty matrix
        matrix = np.zeros((nrows, ncols), dtype=int)
        # Populate the matrix with the solution
        for i in range(nrows):
            for j in range(ncols):
                matrix[i][j] = pulp.value(choices[i][j])
        return matrix
    else:
        return None


def parallel_generate_matrices(n, X, Y, seed=42, max_workers=8):
    """
    Generate n matrices in parallel with different seeds for each.

    Parameters:
    - n: The number of matrices to generate.
    - X: The row sums for the matrices.
    - Y: The column sums for the matrices.

    Returns:
    - A list of generated matrices.
    """
    matrices = {}

    random_gen = random.Random(seed)

    # Use a ThreadPoolExecutor to parallelize tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_binary_matrix, X, Y, seed=random_gen.randrange(2**31 - 1)): i for i in range(n)
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            idx = futures[future]
            try:
                res = future.result()
                print(f"Generated matrix {i + 1}/{n}")
                if res is not None:
                    matrices[idx] = res
            except Exception as e:
                print(f"Matrix generation failed: {e}")

    return [matrices[i] for i in range(n) if i in matrices]


def matrices_to_coo(matrices):
    """
    Convert a list of matrices to the COO format.

    Parameters:
    - matrices: A list of np.ndarray matrices.

    Returns:
    - A DataFrame containing the COO representation with columns ['Matrix', 'Row', 'Col'].
    """
    coo_data = []
    for index, matrix in enumerate(matrices):
        rows, cols = np.where(matrix == 1)
        for row, col in zip(rows, cols):
            coo_data.append([index+1, row, col])  # index+1 to make matrix numbering start from 1

    coo_df = pd.DataFrame(coo_data, columns=['Matrix', 'Row', 'Col'])
    return coo_df


@click.command()
@click.option('--n_matrices', default=5, help='Number of matrices to generate.')
@click.option('--rows', prompt='Enter row sums separated by commas', help='Row sums for the matrices.')
@click.option('--columns', prompt='Enter column sums separated by commas', help='Column sums for the matrices.')
@click.option('--output', default='matrices_coo.csv', help='Output CSV file name.')
@click.option('--seed', default=42, help='Random seed for the generator.')
@click.option('--max_workers', default=8, help='Maximum number of workers for parallel generation.')
def generate_matrices_cli(n_matrices, rows, columns, output, seed, max_workers):
    """
    CLI tool for generating binary matrices with given row and column sums,
    converts them to COO format, and saves as a CSV.
    """
    X = [int(x.strip()) for x in rows.split(',')]
    Y = [int(y.strip()) for y in columns.split(',')]

    generated_matrices = parallel_generate_matrices(n_matrices, X, Y, seed, max_workers)
    coo_df = matrices_to_coo(generated_matrices)

    # Save the COO formatted data to CSV
    coo_df.to_csv(output, index=False)
    click.echo(f"Saved COO formatted matrices to {output}")

if __name__ == '__main__':
    generate_matrices_cli()
