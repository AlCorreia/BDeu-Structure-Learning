from gobnil import *
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import networkx as nx
import sys
from tqdm import tqdm


if __name__ == '__main__':
    dataset = sys.argv[1]
    sample_size = int(sys.argv[2])
    n_runs = int(sys.argv[3])
    score = sys.argv[4]
    alpha = float(sys.argv[5])
    palim = int(sys.argv[6])

    # Read the BN model
    reader = BIFReader(dataset + '.bif')
    model = reader.get_model()
    arities = dict(model.get_cardinality())
    for key, value in arities.items():
        arities[key] = [value]
    nodes = list(nx.topological_sort(model))

    for run in tqdm(range(n_runs)):
        data = pd.read_csv(os.path.join(dataset,
                                        str(sample_size)+'_'+str(run)+'.csv'),
                           sep=' ')
        data = data.iloc[1:]  # Remove the arities from the table
        if score == 'BDeu':
            save_folder = os.path.join(dataset, str(alpha)+'_'+score+'_'+str(sample_size)+'_'+str(run))
        else:
            save_folder = os.path.join(dataset, score+'_'+str(sample_size)+'_'+str(run))
        BN = GobnilPy(nodes, arities)
        BN.learn_structure(data, score, alpha, palim=palim, save=save_folder)
