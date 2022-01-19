import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

epsilon = 0.001


class EntityGraph(object):
    def __init__(self):
        self.nx_graph = nx.Graph()

    def build_graph_entities(self, entities, entities_frequencies, keep_edge_percentage, x=None,
                             similarity_matrix=None):
        """ builds an entity graph given the similarity matrix and entities """
        # add categories nodes and edges
        if x is None and similarity_matrix is None or x is not None and similarity_matrix is not None:
            raise Exception('Argument exception - build_graph_entities in Modularity Maximization -'
                            ' should provide either X or sim(X).')

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
