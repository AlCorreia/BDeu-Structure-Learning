#!/usr/bin/env python3
import argparse
from collections import Counter
from itertools import combinations
from math import lgamma, log, factorial
import numpy as np
import operator
import os
import pandas as pd
from functools import reduce
import sys
import time
import warnings


###############################
##### AUXILIARY FUNCTIONS #####
###############################

def nCr(n, r):
    """
        Returns the number of combinations of r elements out a total of n.
    """
    f = factorial
    return f(n) // f(r) // f(n-r)


def lg(n,x):
    return lgamma(n+x) - lgamma(x)


def onepositive(dist):
    """
        True if there only one positive count in `dist`.
    """
    npos = 0
    for n in dist:
        if n > 0:
            npos += 1
            if npos > 1:
                return False
    return True


def ml(dist):
    """
        Compute the maximum likelihood given full instantiation counts in dist.
    """
    tot = sum(dist)
    res = 0.0
    for n in dist:
        if n > 0:
            res += n*log(n/tot)
    return res


def ml_sum(distss):
    """
        Compute the total maximum likelihood of the full instantiation counts
        in distss.
    """
    res = 0.0
    for dists in distss:
        res += sum([ml(d) for d in dists])
    return res


###############################
####### BOUND FUNCTIONS #######
###############################


def diffa(dist, alpha, r):
    """
        Compute the derivative of local-local BDeu score.
    """

    res = 0.0
    for n in dist:
        for i in range(n):
            res += 1.0/(i*r+alpha)
    for i in range(sum(dist)):
        res -= 1.0/(i+alpha)
    return res


def g(dist, aq):
    """
        Compute function g (Lemma 5) for a given full parent isntantiation.

        Parameters
        ----------

        dists: list ints
            Counts of the child variable for a given full parent instantiation.
        aq: float
            Equivalent sample size divided by the product of parents arities.

    """

    res = log(2*min(dist)/aq + 1)
    for d in dist:
        res += - log(2*d/aq + 1)
    return res


def h(dist, alpha, r):
    """
        Computes function h (Lemma 8).
    """

    res = -lg(sum(dist), alpha)
    alphar = alpha/r
    for n in dist:
        if n > 0:
            res += lg(n, alphar)
    return res


def ubh_js(dists, alpha, r, counters=None):
    """
        Compute bound h for each instantiation js of set of parents S.
        See Theorem 3 for the definition of the bound.

        Parameters
        ----------

        dists: list of lists
            Counts of the child variable for each full parent instantiation j
            that is compatible with js.
        alpha: float
            The equivalent sample size (ESS).
        r: int
            Arity (number of possible values) of the child variable.
        counters: dict (optional)
            Dictionary used to store the number of times each function
            (ml, f + g, h) was the minimum in the last part of the equation in
            Theorem 3.

        Returns
        -------

        Upper bound h for a given isntantiation of parent set S.

    """
    is_g, is_h, is_ml = 0, 0, 1
    mls = 0.0
    best_diff = 0.0
    for dist in dists:
        ml_j = ml(dist)
        mls += ml_j
        ubg_plus_f = -len(dist)*log(r) + g(dist, alpha)
        iffirst_ub = min(ubg_plus_f, ml_j)
        ubh = 0
        if not onepositive(dist) and diffa(dist, alpha, r) >= 0 and alpha <= 1:
            ubh = h(dist, alpha/2, r)
            iffirst_ub = min(iffirst_ub, ubh)

        diff = iffirst_ub - ml_j
        if diff < best_diff:
            best_diff = diff
            is_g, is_h, is_ml = 0, 0, 0
            if iffirst_ub == ubg_plus_f:
                is_g = 1
            if iffirst_ub == ubh:
                is_h = 1

    if counters is not None:
        counters['inner_ml'] += is_ml
        counters['inner_g'] += is_g
        counters['inner_h'] += is_h
        counters['inner_total'] += 1

    return best_diff + mls


###############################
######### DATA CLASS ##########
     ###  main code   ###
###############################


class Data:
    """
    A dataset of complete discrete data.
    This class holds all the information used during the experiments.
    """

    def __init__(self, data, name):
        """"
            Attributes
            ----------

            data: pandas dataframe or path to csv file.
                The data to be used for the experiments.
            name: str
                Name used to save results (usually matching dataset name).

            It is assumed that:

            1. All values are separated by whitespace
            2. Comment lines start with a '#'
            3. The first line is a header stating the names of the variables
            4. The second line states the arities of the variables
            5. All other lines contain the actual data

        """

        if isinstance(data, pd.DataFrame) == False:
            data = pd.read_csv(data,
                               delim_whitespace=True,
                               comment='#')

        arities = [int(x) for x in data.iloc[0]]
        self._name = name
        self._data = data[1:]
        self._arities = dict(zip(list(self._data), arities))
        self._variables = list(data.columns)
        self._varidx = {}
        # Initialize all the counters to zero
        self.counters = {}
        self.reset_counters()

        for i, v in enumerate(self._variables):
            self._varidx[v] = i
        self.get_atoms()

    def reset_counters(self):
        """
            There are a number of counters to keep track of the number of
            scores and bounds computed. This function resets them to zero.
        """
        # 'min' counters are used to keep track the number of times each of
        # bound g and h is the tightest.
        self.counters['min_ubg'] = 0
        self.counters['min_ubh'] = 0
        # 'inner' counters are used inside bound h. See ubh_js function in
        # utils.py or Theorem 3 in the paper.
        self.counters['inner_ml'] = 0
        self.counters['inner_g'] = 0
        self.counters['inner_h'] = 0
        self.counters['inner_total'] = 0

    def upper_bound_f(self, child, posfamilyinsts):
        """
            Compute a weak upper bound on supersets of a given parent set.

            Parameters
            ----------

            child: int
                Index of the child of the family.
            posfamilyinsts: int
                The number of instantiations of the family which occur at least
                once in the data

            Returns
            -------

            Upper bound h (float).

        """
        return -posfamilyinsts * log(self._arities[child])

    def upper_bound_g(self, child, parents, aq, posfamilyinsts, atoms_for_parents):
        """
            Compute an upper bound on supersets of parents

            Parameters
            ----------

            child: int
                Index of the child of the family.
            parents: list
                The parents of the family (an iterable of indices)
            aq: float
                Equivalent sample size divided by the product of parents arities.
            posfamilyinsts: int
                The number of instantiations of the family which occur at least
                once in the data
            atoms_for_parents: dict
                For each instantiation of `parents` (keys in the dictionary), a
                list of list of counts the child variable. Each of the inner
                lists corresponds to a full instantion of all variables (but
                the child) that is compatible with the instantiation of the
                parents in the key. See atoms_for_parents function.

            Returns
            -------

            Upper bound g (float).

        """
        m_final = 0
        for dists in atoms_for_parents:
            pens = []
            # Compute g for each full instantiation.
            for dist in dists:
                pens.append(g(dist, aq))
            m_min = min(pens)
            m_final += m_min
            if len(pens) > 1:
                pens[pens.index(m_min)] = float('inf')
                m_final += min(pens)
        return -posfamilyinsts*log(self._arities[child]) + m_final

    def upper_bound_h(self, child, parents, alpha, atoms_for_parents):
        """
            Compute an upper bound on supersets of parents.

            Parameters
            ----------

            child: int
                Index of the child of the family.
            parents: list
                The parents of the family (an iterable of indices)
            alpha: float
                Equivalent sample size.
            atoms_for_parents: dict
                For each instantiation of `parents` (keys in the dictionary), a
                list of list of counts the child variable. Each of the inner
                lists corresponds to a full instantion of all variables (but
                the child) that is compatible with the instantiation of the
                parents in the key. See atoms_for_parents function.

            Returns
            -------

            Upper bound h (float).

        """

        for pa in parents:
            alpha /= self._arities[pa]
        r = self._arities[child]
        this_ub = 0.0
        # Compute ubh for each instantiation of parent set S
        for dists in atoms_for_parents:
            this_ub += ubh_js(dists, alpha, r, self.counters)
        return this_ub

    def upper_bound_min_min(self, child, parents, aq, counts, atoms_for_parents):
        """
            Returns the best (min) of the two bounds (g and h).

            Parameters
            ----------

            child: int
                Index of the child of the family.
            parents: list
                The parents of the family (an iterable of indices)
            aq: float
                Equivalent sample size divided by the product of parents arities.
            counts: pandas series
                The counts for each of the full instantiations.
                (Only the number of full instantations is actually needed).
            atoms_for_parents: dict
                For each instantiation of `parents` (keys in the dictionary), a
                list of list of counts the child variable. Each of the inner
                lists corresponds to a full instantion of all variables (but
                the child) that is compatible with the instantiation of the
                parents in the key. See atoms_for_parents function.

            Returns
            -------

            Upper bound min(g, h) (float).
        """

        r = self._arities[child]
        this_ub = 0.0
        m_final = 0
        for child_counts in atoms_for_parents:
            # Upper bound h
            this_ub += ubh_js(child_counts, aq, r)
            # Upper bound g
            pens = []
            for cc in child_counts:
                pen = + log(2*min(cc)/aq + 1)
                for c in cc:
                    pen += - log(2*c/aq + 1)
                pens.append(pen)
            m_min = min(pens)
            m_final += m_min
            if len(pens) > 1:
                pens[pens.index(m_min)] = float('inf')
                m_final += min(pens)

        ubg = -len(counts)*log(self._arities[child]) + m_final

        if this_ub < ubg:
            self.counters['min_ubh'] += 1
        elif this_ub > ubg:
            self.counters['min_ubg'] += 1
        else:
            self.counters['min_ubh'] += 1
            self.counters['min_ubg'] += 1

        return min(this_ub, -len(counts)*log(self._arities[child]) + m_final)

    def bdeu_score(self, child, parents, alpha=None, bound=None):
        """
            Computes the (local) score of a given child and a parent set.

            Parameters
            ----------

            child: int
                Index of the child of the family.
            parents: list
                The parents of the family (an iterable of indices)
            alpha: float
                Equivalent sample size.

            Returns
            -------
            A tuple (score, ubs) where
            - score is the BDeu score of a particular child and parent set
            - ubs is a dictionary of mapping the names of upper bounds to
              upper bounds on the BDeu scores of supersets of the parent set.

        """

        if alpha is None:
            alpha = 1.0
            warnings.warn('ESS (alpha) not defined. Defaulting to alpha=1.0.')

        aq = alpha
        for parent in parents:
            aq /= self._arities[parent]
        aqr = aq / self._arities[child]

        counts = self._data.groupby(list(parents)+[child], sort=True).size()
        posfamilyinsts = len(counts)
        bdeu_score = 0.0
        if len(parents) == 0:
            nij = 0
            for nijk in counts:
                bdeu_score += lg(nijk,aqr)
                nij += nijk
            bdeu_score -= lg(nij,aq)
        else:
            cnt = Counter()
            for idx, nijk in counts.iteritems():
                cnt[idx[:-1]] += nijk
                bdeu_score += lg(nijk,aqr)
            for nij in cnt.values():
                bdeu_score -= lg(nij,aq)

        atoms_for_parents = self.atoms_for_parents(child, parents).values()

        if bound == 'f':
            bounds = {'f': self.upper_bound_f(child, posfamilyinsts)}
        elif bound == 'g':
            bounds = {'g': self.upper_bound_g(child, parents, aq, posfamilyinsts, atoms_for_parents)}
        elif bound == 'h':
            bounds = {'h': self.upper_bound_h(child, parents, alpha, atoms_for_parents)}
        elif bound == 'min':
            bounds = {'min': self.upper_bound_min_min(child, parents, aq, counts, atoms_for_parents)}
        elif bound == 'all':
            bounds = {'f': self.upper_bound_f(child, posfamilyinsts),
                      'g': self.upper_bound_g(child, parents, aq, posfamilyinsts, atoms_for_parents),
                      'h': self.upper_bound_h(child, parents, alpha, atoms_for_parents),
                      'min': self.upper_bound_min_min(child, parents, aq, counts, atoms_for_parents)}
        elif bound is None:
            return bdeu_score
        return bdeu_score, bounds

    def pen_ll_score(self, child, parents, pen_type):
        """
        Returns a the AIC score of a particular child and parent set

        Parameters
        ----------

        child: int
            Index of the child of the family.
        parents: list
            The parents of the family (an iterable of indices)
        pen_type: str or float
            Either a type of score ('BIC' or 'AIC') or a penalisation
            coefficient.
        """

        counts = self._data.groupby(list(parents)+[child],sort=True).size()
        posfamilyinsts = len(counts)

        LL = 0
        if len(parents) == 0:
            nij = counts.sum()
            for nijk in counts:
                LL += nijk*np.log(nijk/nij)
            pen = (self._arities[child] -1)
        else:
            cnt = Counter()
            # Compute nij for each parent configuration
            for idx, nijk in counts.iteritems():
                cnt[idx[:-1]] += nijk
            # Compute the loglikelihood
            for idx, nijk in counts.iteritems():
                LL += nijk*np.log(nijk/cnt[idx[:-1]])
            # Compute the penalization for AIC
            pen = 1
            for parent in parents:
                pen *= self._arities[parent]
            pen *= self._arities[child] -1
        if pen_type == 'AIC':
            score = LL - pen
        elif pen_type == 'BIC':
            pen *= 0.5*np.log(counts.sum())
            score = LL - pen
        elif isinstance(pen_type, (int, float)):
            score = LL - pen_type*pen
        else:
            Exception(pen_type + ' is not supported yet. Please use BIC or AIC.')
        return score

    def all_bdeu_scores(self, alpha=None, palim=None, bound=None, filepath=None):
        """
        Exhaustively compute all BDeu scores and upper bounds for all families
        up to `palim`

        Parameters
        ----------
        child: int
            Index of the child of the family.
        alpha: float
            Equivalent sample size.
        palim: int
            The maximum number of parents.
        bound: str
            The bound to compute. Either 'f', 'g', 'h', 'min'.
            If bound == 'all' computes all bounds.
        filepath: str
            Path to file where to save the scores. If left to None, the scores
            are not saved.

        Returns
        -------

        score_dict: dict
            A dictionary dkt where dkt[child][parents] = bdeu_score
        """

        if palim is None:
            palim = 3
            warnings.warn('Maximum number of parents (palim) not defined. Defaulting to palim=3.')
        if alpha is None:
            alpha = 1.0
            warnings.warn('ESS (alpha) not defined. Defaulting to alpha=1.0.')
        score_dict = {}
        for i, child in enumerate(self._variables):
            potential_parents = frozenset(self._variables[:i]+self._variables[i+1:])
            child_dkt = {}
            for pasize in range(palim+1):
                for parents in combinations(potential_parents,pasize):
                    child_dkt[frozenset(parents)] = self.bdeu_score(child,parents,alpha,bound=bound)
            score_dict[child] = child_dkt

        if filepath is not None:
            self.write_scores(filepath, score_dict)
        return score_dict

    def all_pen_ll_scores(self, score_type, filepath=None, palim=None):
        """
        Exhaustively compute all BDeu scores and upper bounds for all families
        up to `palim`

        Parameters
        ----------
        score_type: str or float
            Either a type of score ('BIC' or 'AIC') or a penalisation
            coefficient.
        filepath: str
            Path to file where to save the scores. If left to None, the scores
            are not saved.
        palim: int
            Maximum number of parents.

        Returns
        -------

        score_dict: dict
            A dictionary dkt where dkt[child][parents] = bdeu_score
        """

        if palim is None:
            palim = 3
            warnings.warn('Maximum number of parents (palim) not defined. Defaulting to palim=3.')
        score_dict = {}
        for i, child in enumerate(self._variables):
            potential_parents = frozenset(self._variables[:i]+self._variables[i+1:])
            child_dkt = {}
            for pasize in range(palim+1):
                for parents in combinations(potential_parents,pasize):
                    child_dkt[frozenset(parents)] = self.pen_ll_score(child, parents, score_type)
            score_dict[child] = child_dkt

        if filepath is not None:
            self.write_scores(filepath, score_dict)
        return score_dict

    def write_scores(self, filepath, score_dict):
        """
            Saves a dictionary of scores to filepath.
            See all_pen_ll_scores or all_bdeu_scores.
        """
        score_info = '{}\n'.format(len(self._variables))
        for child, parents in score_dict.items():
            score_info += child + ' {}\n'.format(len(score_dict[child]))
            for parent, score in parents.items():
                score_info += str(score) + ' ' + str(len(parent)) + ' ' + ' '.join(parent) + '\n'
        with open(filepath, 'w') as w:
            w.write(score_info)

    def atoms_for_parents(self, child, parents):
        """
            Return a dictionary whose keys are instantiations of `parents`
            with positive counts in the data and whose values are lists of lists
            of child counts.

            Parameters
            ----------
            child: int
                The (index of) child variable.
            parents: list of ints
                The list of indices of the parent variables.

            Returns
            -------
            dkt: dict
                [parents instantiations] = [[child counts full intantiation 1],
                                            [child counts full intantiation 2],
                                            ...
                                            [child_counts full intantiation n]]
            Example
            -------

            If dkt is the returned dictionary and dkt[0,1,0] = [[1,2], [0,4]],
            then there are 3 variables in `parents` and there are 2 full parent
            instantiations for the instantiation (0,1,0): one with child counts
            [1,2] and one with child counts [0,4].

            A full instantiation means that all variables (but the child) have
            a value assigned to them. The full instantatiations in each key are
            the ones compatible with the corresponding instantiation of `parents`
            in that key. In the example, if we have 4 variables (plus the child)
            that means there are two possible instantiation of the 4th variable:
            one where the child is distributed as [1, 2], and other where it is
            distributed as [0, 4]. The 4th variable might have more than 2
            states, but those are not observed (zero counts) in this example.
        """

        # Get indices of parents in vector of all possible parents for child
        child_idx = self._varidx[child]
        parentset = frozenset(parents)
        pa_idxs = []
        for i, parent in enumerate(self._variables[:child_idx]+self._variables[child_idx+1:]):
            if parent in parentset:
                pa_idxs.append(i)

        # As we open the tree of variables following the index order, we only
        # look at full instantations of parents and variables of higher index.
        upper_pa_idxs = list(range(max(pa_idxs + [-1]) + 1, len(self._variables)-1))
        upper_dkt = {}
        for fullinst, childcounts in self._atoms[child].items():
            inst = tuple([fullinst[i] for i in pa_idxs + upper_pa_idxs])
            try:
                upper_dkt[inst] = list(np.array(upper_dkt[inst]) + np.array(childcounts))
            except KeyError:
                upper_dkt[inst] = childcounts

        # The counts for instantations that differ only on variables of lower
        # index can be safely summed to improve the bounds.
        dkt = {}
        posfamilyinsts = 0
        for fullinst, childcounts in upper_dkt.items():
            inst = tuple([fullinst[i] for i in range(len(pa_idxs))])
            # In this step, we have to remove the zero counts!
            non_zeros = [x for x in childcounts if x>0]
            posfamilyinsts += len(non_zeros)
            try:
                dkt[inst].append(non_zeros)
            except KeyError:
                dkt[inst] = [non_zeros]
        return dkt

    def get_atoms(self):
        """
        Compute a dictionary whose keys are child variables and whose values
        are dictionaries mapping instantiations of all the other parents to a
        list of counts for the child variable for that instantiation.
        Only parent set instantations with a positive count in the data are
        included.

        The dictionary is stored as the value of self._atoms
        """
        # Create the counts as a pandas DataFrame with a new column 'counts'
        counts = pd.DataFrame({'counts' : self._data.groupby(self._variables).size()}).reset_index()

        # Save the counts inside a list to facilitate concatenation
        listfy = lambda x : [x]
        counts['counts'] = counts['counts'].apply(listfy)

        dktall = {}
        for child in self._variables:
            all_but_child = [var for var in self._variables if var != child]
            # The sum operation concatenate the lists of counts
            # for rows that differ only on the child variable
            # The unstack operation fill in the full instantations
            # which do not have all possible values of child in the data
            # so that we can keep the zeros in place
            child_counts = counts.groupby(by=self._variables).agg({'counts': 'sum'}).unstack(child, fill_value=[0]).stack().reset_index()
            child_counts = child_counts.groupby(by=all_but_child).agg({'counts': 'sum'}).reset_index()
            dkt_child = child_counts.set_index(all_but_child).to_dict('index')
            for cc in dkt_child:
                dkt_child[cc] = dkt_child[cc]['counts']

            dktall[child] = dkt_child

        self._atoms = dktall

    def pruned_bdeu_scores_per_child(self, child, bound, timeout, alpha=1.0, palim=None, verbose=False, save_step=False):
        """
        Return a dictionary for the child variable mapping parent sets to
        BDeu scores.

        Not all parent sets are included. Only those parent set of cardinality
        at most `palim` can be included. Also, if it can be established that
        a parent set can not be a parent set for the child in an optimal Bayesian
        network, then it is not included.

        Also, outputs a pandas DataFrame with the number of scores computed.
        The DataFrame is saved to memory every iteration over palim so not to
        miss results if the process is terminated.

        Parameters
        ----------

        child: int
            The (index of) child variable.
        bound: str
            The type of bound to use.
        timeout: int
            The maximum amount of time the function has to run (secs).
        alpha: float
            The effective sample size (prior parameter).
        palim: int
            The maximum number of parents.
        verbose: boolean
            Whether messages on progress should be printed.
        save_step: boolean
            Whether to save a csv per child
        """

        if palim is None:
            palim = 3
            warnings.warn('Maximum number of parents (palim) not defined. Defaulting to palim=3.')
        if bound == 'h':
            scores_per_palim = pd.DataFrame(columns=['child', 'palim', 'alpha', 'all_scores', 'n_scores_' + bound, 'time_' + bound, 'inf_n_scores', 'inner_ml', 'inner_g', 'inner_h', 'inner_total', 'best_pa'])
        elif bound == 'min':
            scores_per_palim = pd.DataFrame(columns=['child', 'palim', 'alpha', 'all_scores', 'n_scores_' + bound, 'time_' + bound, 'inf_n_scores', 'min_ubg', 'min_ubh', 'best_pa'])
        else:
            scores_per_palim = pd.DataFrame(columns=['child', 'palim', 'alpha', 'all_scores', 'n_scores_' + bound, 'time_' + bound, 'inf_n_scores', 'best_pa'])
        p = len(self._variables)
        palim = min(palim, p-1)
        child_idx = self._variables.index(child)

        n_scores = 1  # number of times we calculate the score
        unnecessary_scores = 0  #  counts how many scores were unnecessary
        all_scores = 1  # all combinations of parents
        self.reset_counters()

        score, ubdkt = self.bdeu_score(child, [], alpha, bound=bound)
        ub = min(ubdkt.values())  # take the best bound
        child_dkt = {(): score}
        previous_layer = {(): (score, True) }
        best_score = score
        best_pa = frozenset()

        if not os.path.exists(self._name):
            os.makedirs(self._name)

        timeout = timeout  # convert to secs
        start = time.time()
        for pasize in range(palim):
            new_layer = {}
            all_scores += nCr(len(self._variables)-1, pasize+1)  # all combinations with pasize+1 parents
            for oldpa in previous_layer:
                last_idx = -1 if oldpa == () else self._varidx[oldpa[-1]]
                for newpa_idx in range(last_idx+1, p):

                    # We check the time in the innermost loop
                    end = time.time()
                    if end - start > timeout:
                        elapsed_time = end - start
                        print('End of time! Last palim: ', pasize)
                        scores_per_palim.to_csv(self._name + '/' + child + '_' + bound + '.csv', index=False)
                        return child_dkt, scores_per_palim

                    if newpa_idx == child_idx:
                        continue
                    parents = oldpa + (self._variables[newpa_idx], )
                    bss = None
                    if True:
                        # get best score and upper bound - this could be done more efficiently
                        for (parenttmp, scoretmp) in child_dkt.items():
                            if parenttmp < parents:
                                bss = scoretmp if bss is None else max(bss, scoretmp)
                    else:
                        # IF LAYERS ARE COMPLETE, THEN WE CAN DO EFFICIENTLY (THIS PREVIOUS LOOP):
                        for i in range(len(parents)):
                            try:
                                old_score, _ = previous_layer[parents[:i]+parents[i+1:]]
                            except KeyError:
                                if verbose:
                                    print(parents[:i]+parents[i+1:],'is missing')
                                    bss = None
                                    break
                            bss = old_score if bss is None else max(bss, old_score)

                        if bss is None:
                            # some subset is exponentially pruned, so don't score
                            # or: best we can hope for 'parents' is worse than some existing subset
                            # of parents
                            if verbose:
                                print('Prune!', child, parents, bss, lub)
                            continue

                    score, ubdkt = self.bdeu_score(child, parents, alpha, bound=bound)
                    if score > best_score:
                        best_pa = parents
                        best_score = score
                    ub = min(ubdkt.values())  # take the best bound
                    n_scores = n_scores + 1

                    # if the previous layer had a better score
                    # the last computation was unnecessary
                    if (score <= bss):
                        unnecessary_scores += 1

                    if bss >= ub and bss >= score:
                        if verbose:
                            print('Prune!', child, parents, bss, lub)
                        continue

                    new_layer[parents] = (max(score, bss), bss < score)

            child_dkt.update({parents:val[0] for (parents, val) in new_layer.items() if val[1]})
            previous_layer = new_layer

            elapsed_time = time.time() - start
            if bound == 'h':
                scores_per_palim = scores_per_palim.append({'child': child,
                                                            'palim': pasize+1,
                                                            'alpha': alpha,
                                                            'all_scores': all_scores,
                                                            'n_scores_' + bound: n_scores,
                                                            'time_' + bound: elapsed_time,
                                                            'inf_n_scores':  n_scores - unnecessary_scores,
                                                            'inner_ml': self.counters['inner_ml'],
                                                            'inner_g': self.counters['inner_g'],
                                                            'inner_h': self.counters['inner_h'],
                                                            'inner_total': self.counters['inner_total'],
                                                            'best_pa': best_pa},
                                                            ignore_index=True)
            elif bound == 'min':
                scores_per_palim = scores_per_palim.append({'child': child,
                                                            'palim': pasize+1,
                                                            'alpha': alpha,
                                                            'all_scores': all_scores,
                                                            'n_scores_' + bound: n_scores,
                                                            'time_' + bound: elapsed_time,
                                                            'inf_n_scores':  n_scores - unnecessary_scores,
                                                            'min_ubg': self.counters['min_ubg'],
                                                            'min_ubh': self.counters['min_ubh'],
                                                            'best_pa': best_pa},
                                                            ignore_index=True)
            else:
                scores_per_palim = scores_per_palim.append({'child': child,
                                                            'palim': pasize+1,
                                                            'alpha': alpha,
                                                            'all_scores': all_scores,
                                                            'n_scores_' + bound: n_scores,
                                                            'time_' + bound: elapsed_time,
                                                            'inf_n_scores':  n_scores - unnecessary_scores,
                                                            'best_pa': best_pa},
                                                            ignore_index=True)

            if save_step:
                scores_per_palim.to_csv(self._name + '/' + child + '_' + bound + '.csv', index=False)
        return child_dkt, scores_per_palim
