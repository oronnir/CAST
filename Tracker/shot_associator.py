import math
import os
from collections import Counter, OrderedDict
from enum import Enum

from scipy.sparse import dok_matrix
import numpy as np
import networkx as nx
from EvaluationUtils.vision_metrics import CVMetrics
from Tracker.flow_maximizer import FlowMaximizer
from Animator.bbox_grouper_api import CharacterBoundingBox, CharacterDetectionOutput
from Animator.utils import EPS, deserialize_pickle, serialize_pickle


class VertexType(Enum):
    In = 'in'
    Out = 'out'
    Source = 'source'
    Target = 'target'


class EdgeType(Enum):
    Match = 'match'
    Skip = 'skip'
    Enter = 'enter'
    Detect = 'detect'
    Exit = 'exit'


class ShotAssociator:
    MAX_SKIPS = 5
    MAX_TRACKS = 6
    MAX_SIGNIFICANCE_RATIO = .3
    EXP_WEIGHTS = dict(match_similarity=3.5,
                       p_iou=1.5,
                       p_distance=2.,
                       p_time_gap=1.,
                       p_scale=1.5,
                       p_scaled_distance=4.5)

    def __init__(self, detections, sampled_fps, association_pkl, box_to_track=None):
        skip_gap = math.ceil(sampled_fps / 2)
        self.association_pkl = association_pkl
        self.capacities, self.weights, self.source, self.target, self.num_frames, self.edges, self.vertices, self.k = \
            self._tracks_to_network(box_to_track, detections, sampled_fps, skip_gap)
        self.opt_flow = None

    def associate_shot_boxes(self):
        """solve the max-flow-min-cost modelling of MOT"""
        # load if already calculated
        if os.path.isfile(self.association_pkl):
            associations = self.load()
            return associations

        # in case no detections -> no tracks
        if self.capacities is None:
            return []

        # solve the max-flo-min-cost with LP
        # if self.capacities.count_nonzero() >= 5e5:
        #     z, edges, self.opt_flow = FlowMaximizer.network_simplex(self.capacities, self.weights, self.k)
        # else:
        z, edges, self.opt_flow = FlowMaximizer.max_flow_lp_edges(self.capacities, self.weights)

        # find the top k paths
        offline_tracks, paths_weights = self._get_top_paths()
        print(f'found {len(offline_tracks)} paths of (length, weight): [' +
              '; '.join([f'({len(offline_tracks[i])}, {paths_weights[i]})' for i in range(len(offline_tracks))]) + ']')

        # filter insignificant paths
        significant_tracks = ShotAssociator.filter_insignificant_tracks(offline_tracks, paths_weights)

        # verify feasibility and translate back to bounding box tracks
        bbox_tracks = self.__verify_feasibility(significant_tracks)

        # serialize results
        self._serialize(bbox_tracks)
        return bbox_tracks

    @staticmethod
    def filter_insignificant_tracks(offline_tracks, paths_weights):
        significant_paths = []
        if len(offline_tracks) == 0:
            return significant_paths
        lowest_path_cost = min(paths_weights)
        for track, weight in zip(offline_tracks, paths_weights):
            if abs(weight / lowest_path_cost) < ShotAssociator.MAX_SIGNIFICANCE_RATIO:
                continue
            significant_paths.append(track)
        return significant_paths

    @staticmethod
    def _tracks_to_network(shot_box_to_track: dict, detections_out: CharacterDetectionOutput, sampled_fps,
                           skip_gap: int = 0):
        """
        build the trackers and detections into a network using [1]

        G(V,E,C,f): Graph G with edges E, Cost C per [1] definition and a trivial flow function f.

        [1] http://vision.cse.psu.edu/courses/Tracking/vlpr12/lzhang_cvpr08global.pdf
        :return: a network from
        """
        # degenerate case
        if len(detections_out.CharacterBoundingBoxes) == 0 or len(detections_out.CharacterBoundingBoxes) > 1000:
            print('shot has either no detections or too many detections - skipping shot...')
            return None, None, 0, 0, 0, [], [], 0

        # verify boxes order by frame
        ordered_detections_by_frame = sorted(detections_out.CharacterBoundingBoxes,
                                             key=lambda x: (x.KeyFrameIndex, x.IdInFrame))
        detections = OrderedDict([(d.Id, d) for d in ordered_detections_by_frame])

        # estimate the number of tracks w.r.t. to the flow
        if shot_box_to_track is None:
            num_tracks = max(Counter([d.KeyFrameIndex for box_id, d in detections.items()]).values())
            num_detections = len(detections)
        else:
            num_tracks = len(set(shot_box_to_track.values()))
            num_detections = len(shot_box_to_track)

        # source to vertex/ vertex to target
        p_enter = num_tracks / num_detections
        p_exit = p_enter

        # initializations
        start_frame = 1e10
        end_frame = -1
        source_vertex = Vertex(0, -1, -1, VertexType.Source)
        target_vertex = Vertex(-1, -1, -2, VertexType.Target)
        vertex_list = [source_vertex]
        edge_list = []
        edges_to_target = []
        vertex_index = 1
        prev_frame_index = -1
        prev_frame_out_vertices = []
        acc_previous = []
        frame_to_boxes = dict()
        frame_to_out_vertices = dict()

        # support both offline and online followed with offline flows
        box_id_to_track = OrderedDict((b_id, b_id) for b_id in detections) \
            if shot_box_to_track is None \
            else shot_box_to_track

        # loop the detections and build a network for offline MOT
        for box_id, track_id in box_id_to_track.items():
            box = detections[box_id]
            frame_index = box.KeyFrameIndex
            if prev_frame_index != frame_index:
                prev_frame_index = frame_index
                prev_frame_out_vertices = acc_previous.copy()
                acc_previous = []
                frame_to_boxes[frame_index] = [box_id]
                frame_to_out_vertices[frame_index] = []
            else:
                frame_to_boxes[frame_index].append(box_id)

            # add in vertex
            current_in_vertex = Vertex(vertex_index, track_id, frame_index, VertexType.In, box)
            vertex_list.append(current_in_vertex)
            vertex_index += 1

            # add source edge
            source_edge = Edge(source_vertex.index, current_in_vertex.index, -math.log2(p_enter), EdgeType.Enter)
            edge_list.append(source_edge)

            # add edges from last frame
            for prev_out_vertex in prev_frame_out_vertices:
                match_cost = ShotAssociator._match_weight(prev_out_vertex.box, box, detections_out.NativeKeyframeWidth,
                                                          sampled_fps)
                match_edge = Edge(prev_out_vertex.index, current_in_vertex.index, match_cost, EdgeType.Match)
                edge_list.append(match_edge)

            # add out vertex
            current_out_vertex = Vertex(vertex_index, track_id, frame_index, VertexType.Out, box)
            vertex_list.append(current_out_vertex)
            frame_to_out_vertices[frame_index].append(current_out_vertex)
            vertex_index += 1

            # add in-out edge
            in_out_weight = math.log2((1 - current_in_vertex.box.Confidence) / current_in_vertex.box.Confidence)
            in_out_edge = Edge(current_in_vertex.index, current_out_vertex.index, in_out_weight, EdgeType.Detect)
            edge_list.append(in_out_edge)

            # add target edge
            edges_to_target.append(current_out_vertex)

            # skip connections
            edge_list = ShotAssociator._add_skip_connections(edge_list, frame_index, skip_gap, frame_to_out_vertices,
                                                             current_in_vertex, detections_out.NativeKeyframeWidth,
                                                             sampled_fps)

            # add as next.prev detections
            acc_previous.append(current_out_vertex)

            # figure out the number of vertices
            start_frame = min(start_frame, frame_index)
            end_frame = max(end_frame, frame_index)

        # add the target vertex
        target_vertex.index = vertex_index
        target_vertex.frame_index = end_frame + 1
        vertex_list.append(target_vertex)

        # add edges to target
        for e_target in edges_to_target:
            target_edge = Edge(e_target.index, target_vertex.index, -math.log2(p_exit), EdgeType.Exit)
            edge_list.append(target_edge)

        # dok matrices
        capacities_dok_net, weight_dok_net = ShotAssociator._to_dok(vertex_list, edge_list)

        num_frames = end_frame - start_frame + 1
        boxes_per_frame = [len(boxes) for boxes in frame_to_boxes.values()] + [1]
        max_tracks_estimate = min(ShotAssociator.MAX_TRACKS, max(boxes_per_frame))
        return capacities_dok_net, weight_dok_net, source_vertex.index, target_vertex.index, num_frames, edge_list, \
            vertex_list, max_tracks_estimate

    @staticmethod
    def _match_weight(from_box: CharacterBoundingBox, to_box: CharacterBoundingBox, frame_width: int,
                      sampled_fps: float) -> float:
        """estimate the probability of a match into a weight"""
        # iou
        match_iou = CVMetrics.bb_intersection_over_union(from_box.Rect, to_box.Rect)
        p_iou = np.mean([1., match_iou])

        # scale
        from_area = from_box.Rect.area()
        to_area = to_box.Rect.area()
        sqrt_min_area = math.sqrt(min(from_area, to_area))
        sqrt_max_area = math.sqrt(max(from_area, to_area))
        p_scale = sqrt_min_area / sqrt_max_area

        # distance in pixels
        center_distance = ShotAssociator._boxes_centers_distance(from_box, to_box)
        p_distance = 1 - center_distance / frame_width

        # scale-relative distance
        p_scaled_distance = 1 / (1 + center_distance / float(EPS + sqrt_min_area))

        # semantic similarity
        match_similarity = CVMetrics.cosine_similarity(from_box.Features, to_box.Features)

        # time gap
        p_time_gap = ShotAssociator._time_diff_skip_probability(from_box.KeyFrameIndex, to_box.KeyFrameIndex,
                                                                sampled_fps)

        # aggregate by a geometric mean
        probabilities = [match_similarity ** ShotAssociator.EXP_WEIGHTS['match_similarity'],
                         p_iou ** ShotAssociator.EXP_WEIGHTS['p_iou'],
                         p_distance ** ShotAssociator.EXP_WEIGHTS['p_distance'],
                         p_time_gap ** ShotAssociator.EXP_WEIGHTS['p_time_gap'],
                         p_scale ** ShotAssociator.EXP_WEIGHTS['p_scale'],
                         p_scaled_distance ** ShotAssociator.EXP_WEIGHTS['p_scaled_distance']]
        p_match = np.product(probabilities) ** (1 / sum(ShotAssociator.EXP_WEIGHTS.values()))
        match_cost = math.log2(1 - p_match + EPS)
        return match_cost

    @staticmethod
    def _time_diff_skip_probability(from_frame_ind: int, to_frame_ind: int, sampled_fps: float) -> float:
        """time decay for occlusions"""
        delta_t = sampled_fps*(to_frame_ind - from_frame_ind)
        return 0.995 ** (delta_t - 1)

    @staticmethod
    def _boxes_centers_distance(box1: CharacterBoundingBox, box2: CharacterBoundingBox) -> float:
        return box1.Rect.center_distance(box2.Rect)

    def _get_top_paths(self):
        """translate the flow function to offline tracks"""
        if self.opt_flow is None or self.weights is None:
            return [], []
        top_k_paths = []
        weighted_edges = self.opt_flow.multiply(self.weights).todok()
        paths_weights = []
        for k in range(self.k):
            digraph = nx.DiGraph()
            digraph.add_weighted_edges_from((i, j, w) for ((i, j), w) in weighted_edges.items())
            if self.source not in digraph.nodes or self.target not in digraph.nodes:
                continue
            shortest_path = nx.bellman_ford_path(digraph, source=self.source, target=self.target)
            top_k_paths.append(shortest_path)
            weight_acc = 0
            for i in range(1, len(shortest_path)):
                weight_acc += weighted_edges[shortest_path[i - 1], shortest_path[i]]
                weighted_edges[shortest_path[i - 1], shortest_path[i]] = 0
            paths_weights.append(weight_acc)
        return top_k_paths, paths_weights

    def _convert_to_original(self):
        """debug function - translate the flow function to offline tracks"""
        s_t_paths = self.get_all_paths()
        original_paths = list()
        for s_t_path in s_t_paths:
            current_original_path = list()
            acc_weight = 0
            prev_vertex_index = 0
            for vertex_index in s_t_path:
                acc_weight += self.weights[prev_vertex_index, vertex_index]
                prev_vertex_index = vertex_index
                if self.vertices[vertex_index].v_type is VertexType.In:
                    current_original_path.append(self.vertices[vertex_index].box.Id)
            original_paths.append(current_original_path)

            digits = math.ceil(math.log10(len(self.vertices)))
            length = f'{len(current_original_path):{digits}d}'

            print(f'Path length: {length}; weight: {acc_weight:10.4f} s->{current_original_path}->t')
        pairwise_dists = ShotAssociator.consolidate_paths(original_paths)
        return original_paths

    def get_all_paths(self):
        """enumerate all s-t paths in the flow function"""
        g = nx.from_scipy_sparse_matrix(self.opt_flow)
        return nx.all_simple_paths(g, source=self.source, target=self.target)

    @staticmethod
    def consolidate_paths(paths):
        """debug function"""
        n = len(paths)
        pairwise_distances = np.zeros((n, n), dtype=int)
        for i in range(n):
            to_index = min(i + 20, n)
            for j in range(i + 1, to_index):
                edit_distance = ShotAssociator.edit_distance(paths[i], paths[j])
                pairwise_distances[i, j] = edit_distance
                pairwise_distances[j, i] = edit_distance
        return pairwise_distances

    @staticmethod
    def edit_distance(s1, s2):
        m = len(s1) + 1
        n = len(s2) + 1
        i = j = 0
        dp = np.zeros((m, n), dtype=int)
        for i in range(m):
            dp[i, 0] = i
        for j in range(n):
            dp[0, j] = j
        for i in range(1, m):
            for j in range(1, n):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i, j] = min(dp[i, j - 1] + 1, dp[i - 1, j] + 1, dp[i - 1, j - 1] + cost)

        return dp[i, j]

    @staticmethod
    def _add_skip_connections(edge_list, frame_index, skip_gap, frame_to_out_vertices, curr_in_vertex, frame_width,
                              sampled_fps):
        """add an edge between any skip frames for occluded scenarios"""
        if skip_gap <= 0 or frame_index <= 0:  # or frame_index % skip_gap != 0:
            return edge_list

        max_possible_skips = int(frame_index / skip_gap)
        for skip_steps in range(1, 1 + min(ShotAssociator.MAX_SKIPS, max_possible_skips)):
            from_frame_ind = frame_index - skip_gap * skip_steps

            for from_vertex in frame_to_out_vertices.get(from_frame_ind, []):
                match_cost = ShotAssociator._match_weight(from_vertex.box, curr_in_vertex.box, frame_width, sampled_fps)
                if match_cost > 0:
                    continue
                match_edge = Edge(from_vertex.index, curr_in_vertex.index, match_cost, EdgeType.Skip)
                edge_list.append(match_edge)
        return edge_list

    def __verify_feasibility(self, paths):
        id_to_vertex = {v.index: v for v in self.vertices}
        # verify vertex uniqueness between paths
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if len(set(paths[i]) & set(paths[j]) - {self.source, self.target}) > 0:
                    raise Exception('Infeasible paths due to a non-unique solution!')

        # verify capacity 1 per edge in paths
        for path in paths:
            prev = None
            for v in path:
                if prev is not None:
                    if self.capacities[id_to_vertex[prev].index, id_to_vertex[v].index] != 1:
                        raise Exception('Infeasible path due to invalid capacity!')
                prev = v

        # translate vertices into original bounding boxes
        v_ind_to_vertex = {v.index: v for v in self.vertices}

        bbox_tracks = []
        for track_index in range(len(paths)):
            curr_box_track = []
            track_frames = set()
            for v in paths[track_index]:
                if v_ind_to_vertex[v].v_type == VertexType.In:
                    curr_box_track.append(v_ind_to_vertex[v].box.Id)

                    # verify frame uniqueness per path
                    if v_ind_to_vertex[v].frame_index in track_frames:
                        raise Exception(f'Infeasible path, multiple instances of track id per frame: {v}')
                    else:
                        track_frames.add(v_ind_to_vertex[v].frame_index)
            bbox_tracks.append(curr_box_track)
        return bbox_tracks

    @staticmethod
    def _to_dok(vertex_list, edge_list):
        n = len(vertex_list)
        capacities_dok_net = dok_matrix((n, n), dtype=int)
        weight_dok_net = dok_matrix((n, n), dtype=float)
        for e in edge_list:
            capacities_dok_net[e.from_index, e.to_index] = 1
            weight_dok_net[e.from_index, e.to_index] = e.weight
        return capacities_dok_net, weight_dok_net

    def _serialize(self, bbox_tracks):
        serialize_pickle(self.association_pkl, bbox_tracks)

    def load(self):
        return deserialize_pickle(self.association_pkl)


class Vertex:
    def __init__(self, index: int, track_id: int, frame_index: int, v_type: VertexType,
                 box: CharacterBoundingBox = None):
        self.index = index
        self.box = box
        self.track_id = track_id
        self.frame_index = frame_index
        self.v_type = v_type

    def __repr__(self):
        return f'V(index:{self.index}, frame: {self.frame_index}, box_id: {self.box.Id if self.box else "None"}, ' \
               f'type: {self.v_type})'


class Edge:
    def __init__(self, from_index: int, to_index: int, weight: float, e_type: EdgeType):
        self.from_index = from_index
        self.to_index = to_index
        self.capacity = 1
        self.weight = weight
        self.e_type = e_type

    def __repr__(self):
        return f'E(from:{self.from_index}, to: {self.to_index}, w: {self.weight}, type: {self.e_type})'
