import networkx as nx
import numpy as np
import numpy.linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from .utils import eprint, index_to_val_dict

epsilon = 0.001


def upper_tri_indexing(mat):
    """
    Get the upper triangle without the diagonal
    :param mat:
    :return:
    """
    m = mat.shape[0]
    r, c = np.triu_indices(m, 1)
    return mat[r, c]


class EntityGraph(object):
    def __init__(self):
        self.nx_graph = nx.Graph()

    def build_graph_entities(self, entities, entities_frequencies, keep_edge_percentage, x=None, similarity_matrix=None):
        """ builds an entity graph given the similarity matrix and entities """
        # add categories nodes and edges
        if x is None and similarity_matrix is None or x is not None and similarity_matrix is not None:
            raise Exception('Argument exception - build_graph_entities in Modularity Maximization - should provide either X or sim(X).')

        nodes = set()

        for entity in entities:
            nodes.add(entity)
            self.nx_graph.add_node(entity, count=entities_frequencies[entity])

        # return empty graph in case of too few entities
        if len(entities) <= 3:
            return

        list_entities = list(entities)

        # sklearn's cosine similarity
        if similarity_matrix is None:
            similarity_matrix = cosine_similarity(x)

        # get the confidence cutoff, filter, and add as a new edge
        upper_triangle_values = similarity_matrix[np.triu_indices(len(entities), k=1)]
        keep_percentile = np.percentile(upper_triangle_values, keep_edge_percentage)
        np.fill_diagonal(similarity_matrix, 0)
        keep_indices = similarity_matrix < keep_percentile
        similarity_matrix[keep_indices] = 0
        entities_indices = range(len(list_entities))
        for i in entities_indices:
            for j in entities_indices:
                score = similarity_matrix[i, j]
                if score > 0:
                    self.nx_graph.add_edge(list_entities[i], list_entities[j], weight=score)

    def build_graph_categories(self, matrix, my_dict):
        """ builds a category graph given the matrix of jaccard similarity and dict of index to categories """
        length = len(my_dict)
        for category in my_dict:
            self.nx_graph.add_node(my_dict[category])
        for inx in range(length):
            for jnx in range(length):
                if not self.nx_graph.edges.get((my_dict[inx], my_dict[jnx])) and matrix[inx][jnx] > 0:
                    self.nx_graph.add_edge(my_dict[inx], my_dict[jnx], weight=matrix[inx][jnx])

    def partition_graph(self, communities, entities, entities_frequencies):
        """ Given the communities in a graph and entities it partitions the graph of entities """
        for entity in entities:
            self.nx_graph.add_node(entity, count=entities_frequencies[entity], label_graph=communities.get(entity))
        partition = {}
        for node in self.nx_graph.nodes:
            if self.nx_graph.nodes[node]['label_graph'] in partition:
                partition[self.nx_graph.nodes[node]['label_graph']].add(node)
            else:
                partition[self.nx_graph.nodes[node]['label_graph']] = set()
                partition[self.nx_graph.nodes[node]['label_graph']].add(node)
        return partition

    def partition_graph_categories(self, communities, categories_indices_dict):
        """Give the communities in a graph and a dict of indexes to categories partitions the graph"""
        for i in categories_indices_dict:
            self.nx_graph.add_node(categories_indices_dict[i], label_graph=communities.get(categories_indices_dict[i]))
        partition = {}
        for node in self.nx_graph.nodes:
            if self.nx_graph.nodes[node]['label_graph'] in partition:
                partition[self.nx_graph.nodes[node]['label_graph']].add(node)
            else:
                partition[self.nx_graph.nodes[node]['label_graph']] = set()
                partition[self.nx_graph.nodes[node]['label_graph']].add(node)
        return partition

    def save_graph(self, file_name):
        """ Saves the graph to file_name """
        nx.write_gexf(self.nx_graph, file_name)

    @staticmethod
    def recursive_connectivity_markov_analysis(entities, matrix):
        # split by connectivity elements
        g = nx.from_numpy_matrix(matrix)
        if nx.is_connected(g):
            # estimate the stationary distributions
            normalized = (matrix.T / matrix.sum(axis=1)).T
            p_inf = LA.matrix_power(normalized, 100)
            stationary_probabilities = np.clip(p_inf[0, :], 0, 1)
            sorted_probs = zip(entities, stationary_probabilities)
            dict_probs = {x[0]: x[1] for x in sorted_probs}
            return dict_probs
        else:
            eprint("Error: graph not connected in recursive_connectivity_markov_analysis")

    def get_stationary_centralities(self, partition):
        matrix = self.build_matrix_partition(partition)
        centrality = self.recursive_connectivity_markov_analysis(partition, matrix)
        return centrality

    def build_matrix_partition(self, partition):
        partition_length = len(partition)
        index_to_entity = index_to_val_dict(partition)
        matrix = np.zeros([partition_length, partition_length])
        for inx in range(partition_length):
            for jnx in range(partition_length):
                current_edge = (index_to_entity[inx], index_to_entity[jnx])
                if current_edge in self.nx_graph.edges:
                    matrix[inx][jnx] = self.nx_graph.edges[current_edge]['weight']
        return matrix

    @staticmethod
    def compute_centrality_mean_inner_cluster(partitions_categories, categories_to_entities, category_partition,
                                              centrality):
        centrality_mean_inner_cluster = 0.0
        count_centrality_items = 0
        for category in partitions_categories[category_partition]:
            for entity in categories_to_entities[category]:
                centrality_mean_inner_cluster += centrality[entity]
                count_centrality_items += 1
        centrality_mean_inner_cluster /= count_centrality_items
        return centrality_mean_inner_cluster

    @staticmethod
    def compute_centrality_max_inner_cluster(partitions_categories, categories_to_entities, category_partition,
                                             centrality):
        centrality_max_inner_cluster = 0.0
        for category in partitions_categories[category_partition]:
            for entity in categories_to_entities[category]:
                centrality_max_inner_cluster = max(centrality_max_inner_cluster, centrality[entity])

        return centrality_max_inner_cluster

    def count_categories_entities(self, entities):
        return np.sum([self.nx_graph.nodes[entity]["count"] for entity in entities])
