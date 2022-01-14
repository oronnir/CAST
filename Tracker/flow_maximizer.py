import os

from scipy.sparse import dok_matrix
from scipy.optimize import linprog
from Animator.utils import colored
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class FlowMaximizer:
    """
    Finds the max-flow in a network using the EK algorithm

    Example:
            print("Edmonds-Karp algorithm")
            # make a capacity graph
            # node                    s  o  p  q  r  t
            capacities = np.asarray([[0, 3, 3, 0, 0, 0],  # s
                                     [0, 0, 2, 3, 0, 0],  # o
                                     [0, 0, 0, 0, 2, 0],  # p
                                     [0, 0, 0, 0, 4, 2],  # q
                                     [0, 0, 0, 0, 0, 2],  # r
                                     [0, 0, 0, 0, 0, 0]])  # t

            source = 0
            sink = 5
            max_flow_value, flow_function = FlowMaximizer.max_flow(capacities, source, sink)
            print(f"max_flow_value is: {max_flow_value}\nThe flow is:\n{flow_function}")
    """

    def __init__(self):
        pass

    @staticmethod
    def draw_network(graph):
        ax = plt.subplot(111)
        nx.draw_planar(graph, with_labels=True, font_weight='bold')
        plt.close('all')

    @staticmethod
    def weights_to_edge_list(caps: dok_matrix, weights: dok_matrix):
        edges = []
        edge_id = 0
        non_zeros_x, non_zeros_y = caps.nonzero()
        for i, j in zip(non_zeros_x, non_zeros_y):
            if caps[i, j] <= 0:
                continue
            weight = weights[i, j] if weights is not None else 1
            edges.append({'from': i, 'to': j, 'capacity': caps[i, j], 'id': edge_id, 'weight': weight})
            edge_id += 1
        return edges

    @staticmethod
    def network_simplex(caps: dok_matrix, weights: dok_matrix, k_tracks: int):
        """max-flow-min-cost implementation by networkx"""
        n, _ = caps.shape
        s, t = 0, n - 1

        # construct the network
        edges = FlowMaximizer.weights_to_edge_list(caps, weights)
        g = nx.DiGraph()
        g.add_node(s, demand=-k_tracks)
        g.add_node(t, demand=k_tracks)
        for edge in caps.items():
            ((from_node, to_node), capacity) = edge
            if from_node not in g.nodes:
                g.add_node(from_node, demand=0)
            if to_node not in g.nodes:
                g.add_node(to_node, demand=0)
            g.add_edge(from_node, to_node, weight=weights[from_node, to_node], capacity=capacity)

        # solve the flow problem
        flow_cost, flow_dict = nx.network_simplex(g)

        # format the output
        flow_value = dok_matrix(caps.shape, dtype=float)
        for edge in edges:
            edge['x_int'] = edge['x'] = flow_dict[edge['from']][edge['to']]
            flow_value[edge['from'], edge['to']] = edge['x']
        return flow_cost, edges, flow_value

    @staticmethod
    def max_flow_min_cost(network, source, target):
        """networkx implementation for integers weights and capacities"""
        min_cost_flow = nx.max_flow_min_cost(network, source, target)
        min_cost = nx.cost_of_flow(network, min_cost_flow)
        return min_cost_flow, min_cost

    @staticmethod
    def max_flow_lp_edges(caps: dok_matrix, weights: dok_matrix = None):
        """
        Solve a max-flow problem using LP formulation. The first vertex is the source and the last is the target.
        :param weights: the cost of shipping 1 flow unit on each edge
        :param caps: The network capacities
        :return: z^*, x, edge_list, edge_matrix
        """
        n, _ = caps.shape
        s, t = 0, n - 1
        edges = FlowMaximizer.weights_to_edge_list(caps, weights)

        # consider the max flow from t to s as the objective function
        edge_id = len(edges)
        mock_capacity = caps[s, :].sum()
        edges.append({'from': t, 'to': s, 'capacity': mock_capacity, 'id': edge_id, 'weight': 0})
        caps[t, s] = mock_capacity
        edge_id_to_remove = edge_id

        # define the vector c
        n_dec_vars = len(edges)
        c = np.zeros([n_dec_vars], dtype=float)

        if weights is None:
            for edge in edges:
                if edge['from'] == s:
                    c[edge['id']] = -1
        else:
            for edge in edges:
                c[edge['id']] = edge['weight']

        # junction constraints
        A = dok_matrix((n, n_dec_vars), dtype=int)
        for edge in edges:
            # if the edge enters the vertex:
            A[edge['to'], edge['id']] = 1

            # the edge exit the vertex:
            A[edge['from'], edge['id']] = -1

        b = np.zeros([n], dtype=int)
        bounds = [(0, edge['capacity']) for edge in edges]

        # convert from maximization to minimization according to weights (in C)
        res = linprog(c.flatten(), A_eq=A, b_eq=b, bounds=bounds, options={'maxiter': 500, 'sparse': True})

        # post processing
        z = -res.fun if weights is None else res.fun
        x_ints = res.x.round(decimals=3)
        if res.status != 0:
            print(f'{res.message}{os.linesep}')
            return None, None, None
        else:
            print(colored(res.message, 'green'))

        edges = [e for e in edges if e['id'] != edge_id_to_remove]
        flow_value = dok_matrix(caps.shape, dtype=float)
        for edge in edges:
            edge['x_int'] = x_ints[edge['id']]
            edge['x'] = res.x[edge['id']]
            flow_value[edge['from'], edge['to']] = edge['x'].round(decimals=0)
        return z, edges, flow_value

    @staticmethod
    def max_flow_ek(c, s, t):
        """Edmonds-Karp Algorithm"""
        n = c.shape[0]  # C is the capacity matrix
        f = dok_matrix((n, n), dtype=int)
        path = FlowMaximizer.bfs(c - f, s, t)
        while path is not None:
            aug_flow = min(c[u, v] - f[u, v] for u, v in path)
            for u, v in path:
                f[u, v] += aug_flow
                f[v, u] -= aug_flow
            path = FlowMaximizer.bfs(c - f, s, t)

        # find the flow function
        final_flow = f
        final_flow[final_flow < 0] = 0
        return sum(f[s, i] for i in range(n)), final_flow

    @staticmethod
    def bfs(g, s, t):
        """find path by using BFS"""
        queue = [s]
        paths = {s: []}
        if s == t:
            return paths[s]
        while queue:
            u = queue.pop(0)
            for v in range(g.shape[0]):
                if (g[u, v] > 0) and v not in paths:
                    paths[v] = paths[u] + [(u, v)]
                    if v == t:
                        return paths[v]
                    queue.append(v)
        return None


# if __name__ == '__main__':
#     print("Edmonds-Karp algorithm")
#     # make a capacity graph
#     # node                    s  o  p  q  r  t
#     capacities = dok_matrix([[0, 3, 3, 0, 0, 0],  # s
#                              [0, 0, 2, 3, 0, 1],  # o
#                              [0, 0, 0, 0, 2, 0],  # p
#                              [0, 0, 0, 0, 4, 2],  # q
#                              [0, 0, 0, 0, 0, 2],  # r
#                              [0, 0, 0, 0, 0, 0]])  # t
#
#     source_v = 0
#     sink_v = 5
#     max_flow_value, flow_function = FlowMaximizer.max_flow_ek(capacities, source_v, sink_v)
#     print(f"EK: max_flow_value is: {max_flow_value}\nThe flow is:\n{flow_function}")
#     print('\n***\n')
#
#     print('LP Simplex for max flow')
#     total_cost, flow_list, flow_mat = FlowMaximizer.max_flow_lp_edges(capacities)
#     print(f"LP: max_flow_value is: {total_cost}\nThe flow is:\n{flow_mat.toarray()}")
#     print('\n***\n')
#
#     print('LP Simplex for weighted max flow')
#     flow_costs = dok_matrix([[0, 1, 10, 0, 0, 0],  # s
#                              [0, 0, 1, 1, 0, 1],  # o
#                              [0, 0, 0, 0, 1, 0],  # p
#                              [0, 0, 0, 0, 1, 1],  # q
#                              [0, 0, 0, 0, 0, 1],  # r
#                              [0, 0, 0, 0, 0, 0]])  # t
#
#     w_total_cost, w_flow_list, w_flow_mat = FlowMaximizer.max_flow_lp_edges(capacities, weights=flow_costs)
#     print(f"LP: weighted_max_flow_value is: {w_total_cost}\nThe flow is:\n{w_flow_mat.toarray()}")
#     stop = 1
