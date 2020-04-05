#!/usr/bin/env python3

import argparse
import pandas as pd
from functools import reduce
from scoring import Data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='asia1000.csv',
    )

    parser.add_argument(
        '--child', '-c',
        type=str,
        default='all',
    )

    parser.add_argument(
        '--bound', '-b',
        type=str,
        default='all',
    )

    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=1e12,
    )

    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--palim', '-p',
        type=int,
        default=-1,
    )

    FLAGS, unparsed = parser.parse_known_args()

    d = Data('Datasets/' + FLAGS.data, FLAGS.data[:-4])
    if FLAGS.alpha != 1.0:
        # Add prior value to results name
        d._name = str(FLAGS.alpha) + '_' + d._name

    # If palim equals -1, set it to the total number of variables
    # (no limit on the number of parents)
    if FLAGS.palim == -1:
        FLAGS.palim = len(d._variables)

    # Run for all variables and bounds
    if FLAGS.child == 'all' and FLAGS.bound == 'all':
        df_children = []
        for child in d._variables:
            print()
            print((' ' + child + ' ').center(80, '-'))
            df_bounds = []
            for b in  ['f', 'g', 'h', 'min']:
                print('     Computing ' + b + ' bound')
                _, df = d.pruned_bdeu_scores_per_child(child, b, FLAGS.timeout, palim=FLAGS.palim, alpha=FLAGS.alpha)
                df_bounds.append(df)
            df_child = reduce(lambda left, right: pd.merge(left, right, on=['child', 'palim', 'alpha', 'all_scores', 'inf_n_scores', 'best_pa'], suffixes=('_x', '_y')), df_bounds)
            df_children.append(df_child)
        df_final = pd.concat(df_children, axis=0)
        df_final.to_csv(d._name + '/complete.csv', index=False)

    # Run for all variables with a specific bound
    elif FLAGS.child == 'all':
        print('\nRunning for all variables and ' + FLAGS.bound + ' bound on ' + FLAGS.data)
        print(''.center(80, '*'))
        df_children = []
        for child in d._variables:
            print()
            print((' ' + child + ' ').center(80, '-'))
            _, df = d.pruned_bdeu_scores_per_child(child, FLAGS.bound, FLAGS.timeout, palim=FLAGS.palim, alpha=FLAGS.alpha)
            df_children.append(df)
        df_final = pd.concat(df_children, axis=0)
        df_final.to_csv(d._name + '/' + FLAGS.bound + '_' + d._name + '.csv', index=False)

    # Run all bounds for a specific child variable
    elif FLAGS.bound == 'all':
        if FLAGS.child.isdigit():
            try:
                FLAGS.child = d._variables[int(FLAGS.child)]
            except:
                print('The child ID exceeds the total number of variables.')
        elif FLAGS.child not in d._variables:
            raise Exception('Variable %s does not exist.' %(FLAGS.child))

        print('Running for all bounds on ' + FLAGS.child + ' on ' + FLAGS.data)
        df_bounds = []
        for b in ['f', 'g', 'h', 'min']:
            print('     Computing ' + b + ' bound')
            _, df = d.pruned_bdeu_scores_per_child(FLAGS.child, b, FLAGS.timeout, palim=FLAGS.palim, alpha=FLAGS.alpha)
            df_bounds.append(df)
        df_child = reduce(lambda left, right: pd.merge(left, right, on=['child', 'palim', 'all_scores', 'inf_n_scores']), df_bounds)
        df_child.to_csv(d._name + '/' + FLAGS.child + '_all_bounds.csv', index=False)

    # Run for specific child and bound
    else:
        if FLAGS.child.isdigit():
            try:
                FLAGS.child = d._variables[int(FLAGS.child)]
            except:
                print('The child ID exceeds the total number of variables.')
        elif FLAGS.child not in d._variables:
            raise Exception('Variable %s does not exist.' %(FLAGS.child))
        d.pruned_bdeu_scores_per_child(FLAGS.child, FLAGS.bound, FLAGS.timeout, palim=FLAGS.palim, alpha=FLAGS.alpha)
        df_child.to_csv(d._name + '/' + FLAGS.child + '_' + FLAGS.bound + '.csv', index=False)
