"""
PixaMap Vector Engine: Road Centerline & Network Graph Extraction.
Extracts routable 1D topological line graphs (LineString) from 2D raster road corridors,
preserves intersection nodes, prunes short spurs, and calculates road metrics.
"""

from typing import List, Dict, Any, Tuple, Set
import math
import numpy as np
import cv2
import networkx as nx
from shapely.geometry import LineString


class RoadCenterlineExtractor:
    """
    Transforms raster road masks into connected topological vector LineString graphs.
    """

    def __init__(
        self,
        min_road_length_meters: float = 15.0,
        pixel_size_meters: float = 0.05,
        min_confidence_threshold: float = 0.75
    ):
        self.min_road_length_meters = min_road_length_meters
        self.pixel_size_meters = pixel_size_meters
        self.min_confidence_threshold = min_confidence_threshold

    def extract_centerlines(
        self,
        road_prob_mask: np.ndarray,
        geo_transform_fn  # callable: (px, py) -> (lon, lat)
    ) -> List[Dict[str, Any]]:
        """
        Extracts topological road network centerlines from a probability mask.
        """
        # Threshold road mask
        if road_prob_mask.dtype != np.uint8:
            binary_mask = (road_prob_mask >= 0.5).astype(np.uint8) * 255
        else:
            binary_mask = road_prob_mask

        # Step 1: Compute distance transform to estimate road width
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # Step 2: Skeletonization (Morphological Thinning)
        skeleton = self._morphological_thinning(binary_mask)

        # Step 3: Build NetworkX Spatial Graph from 1-pixel skeleton
        graph = self._build_graph_from_skeleton(skeleton)

        if graph.number_of_nodes() == 0:
            return []

        # Step 4: Prune short dead-end spurs (< min_road_length_meters)
        min_nodes = max(3, int(self.min_road_length_meters / max(self.pixel_size_meters, 0.01)))
        graph = self._prune_spurs(graph, min_length_nodes=min_nodes)

        # Step 5: Extract Paths / LineStrings between Junctions (degree != 2)
        edges_paths = self._extract_edge_paths(graph)

        features = []
        feature_id = 1

        for path in edges_paths:
            if len(path) < 2:
                continue

            # Convert pixel coords to geographic CRS coordinates
            geo_coords = []
            widths = []
            probs = []

            for r, c in path:
                gx, gy = geo_transform_fn(float(c), float(r))
                geo_coords.append((gx, gy))
                widths.append(float(dist_transform[r, c]) * 2.0 * self.pixel_size_meters)
                probs.append(float(road_prob_mask[r, c]))

            line = LineString(geo_coords)
            length_px = float(len(path))
            length_m = round(length_px * self.pixel_size_meters, 2)
            avg_width_m = round(float(np.mean(widths)), 2)
            mean_conf = round(float(np.mean(probs)), 3)

            if length_m < self.min_road_length_meters:
                continue

            feature = {
                "type": "Feature",
                "id": feature_id,
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(line.coords)
                },
                "properties": {
                    "feature_type": "road_centerline",
                    "road_id": feature_id,
                    "length_m": length_m,
                    "estimated_width_m": avg_width_m,
                    "confidence_score": mean_conf,
                    "needs_review": bool(mean_conf < self.min_confidence_threshold)
                }
            }
            features.append(feature)
            feature_id += 1

        return features

    def _morphological_thinning(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Zhang-Suen morphological skeletonization algorithm using OpenCV.
        """
        img = (binary_img > 0).astype(np.uint8)
        prev = np.zeros_like(img)
        diff = None

        while True:
            # Sub-iteration 1
            marker1 = self._thinning_iteration(img, 0)
            img = img & ~marker1

            # Sub-iteration 2
            marker2 = self._thinning_iteration(img, 1)
            img = img & ~marker2

            diff = np.abs(img.astype(np.int32) - prev.astype(np.int32))
            prev = img.copy()
            if np.sum(diff) == 0:
                break

        return img.astype(np.uint8) * 255

    def _thinning_iteration(self, im: np.ndarray, iter_num: int) -> np.ndarray:
        marker = np.zeros_like(im)
        h, w = im.shape
        # Pad image to prevent out of bounds
        padded = np.pad(im, 1, mode='constant', constant_values=0)

        # 8-neighbors
        p2 = padded[0:h, 1:w+1]
        p3 = padded[0:h, 2:w+2]
        p4 = padded[1:h+1, 2:w+2]
        p5 = padded[2:h+2, 2:w+2]
        p6 = padded[2:h+2, 1:w+1]
        p7 = padded[2:h+2, 0:w]
        p8 = padded[1:h+1, 0:w]
        p9 = padded[0:h, 0:w]
        p1 = padded[1:h+1, 1:w+1]

        # Condition 1: 2 <= B(P1) <= 6
        bp1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        c1 = (bp1 >= 2) & (bp1 <= 6)

        # Condition 2: A(P1) == 1 (0->1 transitions in clockwise order)
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int) +
            ((p3 == 0) & (p4 == 1)).astype(int) +
            ((p4 == 0) & (p5 == 1)).astype(int) +
            ((p5 == 0) & (p6 == 1)).astype(int) +
            ((p6 == 0) & (p7 == 1)).astype(int) +
            ((p7 == 0) & (p8 == 1)).astype(int) +
            ((p8 == 0) & (p9 == 1)).astype(int) +
            ((p9 == 0) & (p2 == 1)).astype(int)
        )
        c2 = (transitions == 1)

        if iter_num == 0:
            c3 = (p2 * p4 * p6 == 0)
            c4 = (p4 * p6 * p8 == 0)
        else:
            c3 = (p2 * p4 * p8 == 0)
            c4 = (p2 * p6 * p8 == 0)

        to_remove = (p1 == 1) & c1 & c2 & c3 & c4
        marker[to_remove] = 1
        return marker

    def _build_graph_from_skeleton(self, skeleton: np.ndarray) -> nx.Graph:
        """Constructs an undirected graph where white pixels are nodes and 8-neighbors are edges."""
        graph = nx.Graph()
        coords = np.argwhere(skeleton > 0)
        coord_set = set((r, c) for r, c in coords)

        for r, c in coords:
            graph.add_node((r, c))
            # Check 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coord_set:
                        graph.add_edge((r, c), (nr, nc))

        return graph

    def _prune_spurs(self, graph: nx.Graph, min_length_nodes: int) -> nx.Graph:
        """Prunes dead-end spurs shorter than min_length_nodes."""
        pruned = graph.copy()
        while True:
            endpoints = [n for n, deg in pruned.degree() if deg == 1]
            removed = False
            for ep in endpoints:
                # Trace path until junction or other endpoint
                path = [ep]
                curr = ep
                while True:
                    neighbors = [nbr for nbr in pruned.neighbors(curr) if nbr not in path]
                    if not neighbors or pruned.degree(curr) > 2:
                        break
                    curr = neighbors[0]
                    path.append(curr)
                    if pruned.degree(curr) != 2:
                        break
                
                if len(path) < min_length_nodes:
                    pruned.remove_nodes_from(path[:-1])
                    removed = True
            if not removed:
                break
        return pruned

    def _extract_edge_paths(self, graph: nx.Graph) -> List[List[Tuple[int, int]]]:
        """Compresses chains of degree-2 nodes into individual LineString paths."""
        paths = []
        visited_edges = set()

        junctions = [n for n, deg in graph.degree() if deg != 2]
        if not junctions and graph.number_of_nodes() > 0:
            # It's an isolated closed loop or single segment
            nodes = list(graph.nodes())
            return [nodes]

        for j in junctions:
            for nbr in graph.neighbors(j):
                edge = tuple(sorted((j, nbr)))
                if edge in visited_edges:
                    continue

                # Walk path
                path = [j, nbr]
                visited_edges.add(edge)
                curr = nbr

                while graph.degree(curr) == 2:
                    next_nodes = [n for n in graph.neighbors(curr) if n != path[-2]]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    next_edge = tuple(sorted((curr, next_node)))
                    visited_edges.add(next_edge)
                    path.append(next_node)
                    curr = next_node

                paths.append(path)

        return paths
