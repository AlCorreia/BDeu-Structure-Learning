import numpy as np
import os
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from scipy.special import kl_div
from scoring import Data
import shutil
from shutil import copyfile
import subprocess
import tempfile
import warnings


my_env = os.environ.copy()
my_env["PATH"] = "/Users/alvaro/gobnilp/bin:" + my_env["PATH"]
my_env["PATH"] = "/Users/experiments/packs/gobnilp/bin:" + my_env["PATH"]


def create_gobnilp_settings(score_type, palim=None, alpha=None):
    """
    Creates a string of Gobnilp settings given the allowed arities and parent size.

    Parameters
    ----------
    score_type : int
        The scoring function used to learn the network
        0 for BDeu or 2 for BIC
    palim: int
        The maximum number of parents a node can have.
    alpha : float
        The Equivalent Sample Size of the BDeu Score.
    Returns
    -------
    gobnilp_settings : str
        A string describing the provided constraints.
    """

    if palim is None:
        palim = 3
        warnings.warn('Maximum number of parents (palim) not defined. Defaulting to palim=3.')
    if score_type=='BDeu' and alpha is None:
        alpha = 1.0
        warnings.warn('ESS (alpha) not defined. Defaulting to alpha=1.0.')

    output_dot_location = 'sol.mat'

    gobnilp_settings = ''
    gobnilp_settings += 'gobnilp/scoring/palim = {}\n'.format(palim)
    gobnilp_settings += 'gobnilp/scoring/initpalim = {}\n'.format(palim)
    gobnilp_settings += 'gobnilp/delimiter = " "\n'
    gobnilp_settings += 'gobnilp/mergedelimiters = TRUE\n'
    gobnilp_settings += 'gobnilp/outputfile/solution = "golb.bn"\n'
    gobnilp_settings += 'gobnilp/outputfile/adjacencymatrix = "sol.mat"\n'
    gobnilp_settings += 'gobnilp/scoring/alpha = {}\n'.format(format(alpha,'f'))
    gobnilp_settings += 'limits/time = 3600\n'
    if score_type in ["BIC", "BDeu"]:
        gobnilp_settings += 'gobnilp/scoring/score_type = "{}"\n'.format(score_type)

    return gobnilp_settings


def prob(G, datapoint):
    """
    Parameters
    ----------
    G : pgmpy BayesianModel
        A Bayesian network (with learned parameters).
    datapoint : dict
        One observation from the dataset indexed by the variables names.

    Returns
    -------
    prob : float
        The probability of observing that datapoint given the network.
    """

    prob = 1
    for node in G.nodes():
        values = G.get_cpds(node).get_values()
        variables = G.get_cpds(node).variables[1:]
        index_0 = datapoint[node]
        index_1 = 0
        for i, var in enumerate(variables):
            index_1 += pow(2,i)*datapoint[var]
        value = values[index_0][index_1]
        prob = prob * values[index_0][index_1]
    return prob


def log_likelihood(G, data):
    """
    Parameters
    ----------
    G : pgmpy BayesianModel
        A Bayesian network (with learned parameters).
    data : pandas DataFrame
        The data to calculate the likelihood on.

    Returns
    -------
    log_likelihood : float
        The log likelihood of observing the data given the network.
    """

    log_likelihood = 0
    for i in range(data.shape[0]):
        d = data.iloc[i].to_dict()
        log_likelihood += np.log(prob(G, d)+1e-8)
    return log_likelihood


def empirical_kl(P, Q, data):
    """
    Parameters
    ----------
    P : pgmpy BayesianModel
        The target Bayesian network.
    Q : pgmpy BayesianModel
        The learned Bayesian network.
    data : pandas DataFrame
        The data to calculate the likelihood on.
        It should be free of duplicates.
    Returns
    -------
    coverage : float
        A number in (0.0, 1.0).
        The ratio of how much of the whole distribution is covered in data.
    empirical_kl : float
        The empirical KL divergence between the two networks given the data.
    """

    empirical_kl = 0
    coverage = 0
    for i in range(data.shape[0]):
        d = data.iloc[i].to_dict()
        p = prob(P, d)
        q = prob(Q, d)
        empirical_kl += kl_div(p, q)
        coverage += p
    empirical_kl = empirical_kl/coverage
    return coverage, empirical_kl


class GobnilPy(BayesianModel):
    """
    Attributes
    ----------
    G: pgmpy BayesianModel object.
    nodes: list of str
        The names of the variables in the network.
    arities: dict
        The arity of each of the variables in a network.
    """

    def __init__(self, nodes, arities, mat_file=None):
        """

        Parameters
        ----------
        data : pd.DataFrame or str
            The dataframe or holding the data or a path to it.
        arities: dict
            The arity of each of the variables in a network.
        """
        BayesianModel.__init__(self)

        # if isinstance(data, pd.DataFrame):
        #     self.data = data
        # else:
        #     self.data = pd.read_csv(data, sep = ' ')

        self.add_nodes_from(nodes)
        self.arities = arities
        self.state_names = {}
        for node in nodes:
            self.state_names[node] = [x for x in range(arities[node][0])]
        if mat_file is not None:
            self.parse_gobnilp_structure(mat_file)

    def learn_structure(self, data, score, alpha, palim=None, save=None):
        """
        Searches for the best Bayesian network structure given the data
        using Gobnilp https://www.cs.york.ac.uk/aig/sw/gobnilp/

        Parameters
        ----------
        data : pandas DataFrame
            The data to the train the model.
            Only the observations, no arities in the first row.
        score : str
            The scoring function used to learn the network
            'BDeu', 'BIC' or 'AIC'
        palim : int
            The maximum number of parents of each variable.
            A small palim shrinks the search space for Gobnilp, but also
            reduces the chance of finding the correct network.
        alpha : float
            The Equivalent Sample Size of the BDeu Score.
        save : str
            Path to folder where to save the adjacency matrix and the data.

        """
        if palim is None:
            palim = 3
            warnings.warn('Maximum number of parents (palim) not defined. Defaulting to palim=3.')

        if score not in ['AIC', 'BIC', 'BDeu']:
            raise Exception(score + ' is not a valid score type. Please use BDeu, BIC or AIC.')
        settings = create_gobnilp_settings(score, palim, alpha)
        # Create a tempfile to hold the files Gobnilp requires
        folder = tempfile.mkdtemp()
        f = open(os.path.join(folder, 'settings.txt'), 'w')
        try:
            f.write(settings)
        finally:
            f.close()
        # Gobnilp expects the first row to contain the arities of the variables
        data = pd.concat([pd.DataFrame.from_dict(self.arities), data],
                         ignore_index=True,
                         sort=False)
        data.to_csv(os.path.join(folder, 'data.dat'), sep = ' ', index=False)
        if score in ['AIC']:
            d = Data(data, '')
            d.all_pen_ll_scores(filepath=os.path.join(folder, 'scores.txt'), palim=palim, score_type=score)
            proc = subprocess.Popen(['gobnilp -g=settings.txt -f=jkl scores.txt'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=folder,
                                    shell=True,
                                    env=my_env)
        else:
            # Run Gobnilp using a shell command
            proc = subprocess.Popen(['gobnilp -g=settings.txt data.dat'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=folder,
                                    shell=True,
                                    env=my_env)
        proc.wait()
        # Copy files to a different folder
        if save is not None:
            if not os.path.exists(save):
                os.makedirs(save)
            copyfile(os.path.join(folder, 'data.dat'),
                     os.path.join(save, 'data.dat'))
            copyfile(os.path.join(folder, 'settings.txt'),
                     os.path.join(save, 'settings.txt'))
            copyfile(os.path.join(folder, 'sol.mat'),
                     os.path.join(save, 'sol.mat'))
            copyfile(os.path.join(folder, 'golb.bn'),
                     os.path.join(save, 'golb.bn'))
            if os.path.isfile(os.path.join(folder, 'scores.txt')):
                copyfile(os.path.join(folder, 'scores.txt'),
                         os.path.join(save, 'scores.txt'))
        # Open and process the solution
        self.parse_gobnilp_structure(folder + '/sol.mat')
        # Delete temp folder
        shutil.rmtree(folder)

    def learn_parameters(self, data, estimator='Bayesian', alpha=None):
        """
        Learns the parameters of the Bayesian network using pgmpy's built in
        estimators: Bayesian or Maximum Likelihood.

        Parameters
        ----------

        data : pandas DataFrame
            The data to the train the model.
            Only the observations, no arities in the first row.
        estimator: str
            The name of the estimator to be used to fit the data.
            "Bayesian" for BayesianEstimator (default),
            anything else for MaximumLikelihoodEstimator
        """

        if estimator=='Bayesian' and alpha is None:
            alpha = 1.0
            warnings.warn('ESS (alpha) not defined. Defaulting to alpha=1.0.')
        if estimator == 'Bayesian':
            self.fit(data, estimator=BayesianEstimator, equivalent_sample_size=alpha, state_names=self.state_names)
        else:
            self.fit(data)

    def get_all_cpds(self):
        """
        Prints the CPD of each node in the network.
        """
        
        for node in self.nodes():
            print(node)
            print(self.get_cpds(node))


    def parse_gobnilp_structure(self, mat_file):
        """
        Parses an adjacency matrix (mat file) which specifies the structure of a Bayesian Network and transforms it into a PGMPY model.

        Parameters
        ----------

            mat_file : .mat
                The mat file with the adjacency matrix
        """

        # We first add every node so that a node with no edges is not left out
        file = open(mat_file, 'r')
        matrix = np.loadtxt(file)
        for i, A in enumerate(self.nodes()):
            for j, B in enumerate(self.nodes()):
                if matrix[i, j] == 1:
                    self.add_edge(A, B)
        file.close()
