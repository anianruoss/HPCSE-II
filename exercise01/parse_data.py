from collections import OrderedDict
from os import listdir, path

import pandas as pd

data_files = [file for file in listdir('data') if file.endswith('.txt')]

for data_file in data_files:
    parameter = data_file.split(
        'optimal_'
    )[1].split('.txt')[0].title().replace('_', '')

    iterations = []
    times = []
    verifications = []

    with open(path.join('data', data_file), 'r') as f:
        for line in f:
            if line.startswith('V-Cycle Iterations'):
                iterations.append(line.split(': ')[1].rstrip())
            elif line.startswith('Running Time'):
                times.append(line.split(': ')[1].rstrip())
            elif line.startswith('Verification'):
                verifications.append(line.split(' ')[1].split('.')[0])
            else:
                pass

    assert len(iterations) == len(times)
    assert len(times) == len(verifications)

    parameters = list(range(1, len(times) + 1))

    results = pd.DataFrame(
        OrderedDict({
            parameter: parameters,
            'Iterations': iterations,
            'Running Time': times,
            'Verification': verifications
        })
    )

    results.to_latex(
        buf=path.join('data', data_file.replace('txt', 'tex')),
        index=False,
        column_format='c|ccc'
    )
