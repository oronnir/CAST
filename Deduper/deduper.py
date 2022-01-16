import os
import subprocess
import networkx as nx
import numpy as np
from Animator.consolidation_api import CharacterDetectionOutput
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from E2E.configuration_loader import Configuration


config = Configuration().get_configuration()
EXE_PATH = config['deduper']


class PranjaWrapper:
    """
    This module computes Edge Directional Histogram EDH features for duplicates consolidation.

    NOTE: Its application optimizes the downstream clustering, but it is not mandatory for CAST.
    More details are available in the paper.
    """
    def __init__(self):
        self.exe_path = EXE_PATH
        if not os.path.isfile(self.exe_path):
            raise Exception(f'Exe file does not exist at: {self.exe_path}')

    def extract_edh_features(self, detection_output_json, output_json_path):
        if not os.path.isfile(detection_output_json):
            raise Exception(f'Input json does not exist: "{detection_output_json}"')
        exit_code = subprocess.call(self._command_line(detection_output_json, output_json_path))
        if exit_code != 0:
            raise Exception(f'failed extracting keyframes with exit code {exit_code}')
        if not os.path.isfile(output_json_path):
            raise Exception(f'Output json does not exist: "{output_json_path}"')
        return CharacterDetectionOutput.read_from_json(output_json_path)

    def _command_line(self, detection_json, edh_output_json):
        return f'{self.exe_path} extractdupfeature -i "{detection_json}" -f "64D_EDH2x2" -o "{edh_output_json}" -image 0'


class GraphAnalyzer:
    def __init__(self, features):
        graph_csr = GraphAnalyzer.vectors_to_csr(features)
        self.Graph = nx.from_scipy_sparse_matrix(graph_csr)

    def find_cliques(self):
        cliques = nx.algorithms.clique.find_cliques(self.Graph)
        return cliques

    @staticmethod
    def vectors_to_csr(dup_features):
        cosine_sim = cosine_similarity(dup_features)
        n = dup_features.shape[0]
        duplications = csr_matrix((n, n), dtype=np.int8)
        threshold = 0.995
        duplications[cosine_sim > threshold] = 1

        return duplications
