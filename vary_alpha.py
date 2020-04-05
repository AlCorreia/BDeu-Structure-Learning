import argparse
import numpy as np
import pandas as pd
from random import shuffle
from scoring import Data
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='asia1000.csv',
    )

    parser.add_argument(
        '--n_runs', '-n',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=1e12,
    )

    parser.add_argument(
        '--palim', '-p',
        type=int,
        default=-1,
    )

    FLAGS, unparsed = parser.parse_known_args()

    d = Data('Datasets/' + FLAGS.data, FLAGS.data[:-4])
    d._name = 'alphas_' + d._name
    N = d._data.shape[0]
    variables = d._variables
    order = d._variables

    if FLAGS.palim == -1:
        FLAGS.palim=len(d._variables)

    for run in range(FLAGS.n_runs):
        shuffle(order)
        d._variables = order
        for i, v in enumerate(order):
            d._varidx[v] = i

        df_children = []
        for child in order[:-1]:
            print()
            print((' ' + child + ' ').center(80, '-'))
            dfs = []
            for alpha in np.logspace(-10, np.log2(N), num=20, base=2):
                print('     Computing ' + str(alpha))
                _, df = d.pruned_bdeu_scores_per_child(child, 'min', FLAGS.timeout, palim=FLAGS.palim, alpha=alpha)
                # Get only the maximum palim computed
                dfs.append(df[df['palim']==max(df['palim'])])
            df_child = pd.concat(dfs) # reduce(lambda left, right: pd.merge(left, right, on=['child', 'palim']), dfs)
            df_children.append(df_child)
            d._variables = [x for x in d._variables if x != child]

        df_final = pd.concat(df_children, axis=0)
        df_final.to_csv(d._name + '/' + ''.join(order) + '.csv', index=False)
