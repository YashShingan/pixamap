"""
PixaMap Vector Engine: Road Centerline & Network Graph Extraction.
Extracts routable 1D topological line graphs (LineString) from 2D raster road corridors,
preserves intersection nodes, prunes short spurs, and calculates road metrics.
"""

from typing import List, Dict, Any, Tuple, Set, Optional
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
        geo_transform_fn,  # callable: (px, py) -> (lon, lat)
        pixel_size_meters: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts topological road network centerlines from a probability mask.
        """
        px_m = pixel_size_meters if pixel_size_meters is not None else self.pixel_size_meters

        # Threshold road mask to isolate well-defined transportation corridors
        if road_prob_mask.dtype != np.uint8:
            binary_mask = (road_prob_mask >= 0.40).astype(np.uint8) * 255
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

        # Step 4: Prune short dead-end spurs (< 25m on satellite) to eliminate incomplete stubs
        min_spur = max(25.0, self.min_road_length_meters) if px_m >= 0.3 else self.min_road_length_meters
        graph = self._prune_spurs_metric(graph, min_spur_meters=min_spur, pixel_size_meters=px_m)

        # Step 5: Extract Paths / LineStrings between Junctions (degree != 2)
        edges_paths = self._extract_edge_paths(graph)

        features = []
        feature_id = 1

        for raw_path in edges_paths:
            if len(raw_path) < 2:
                continue

            # Apply degree-2 collinear edge collapse (< 15 degrees deviation)
            path = self._collapse_collinear_edges(raw_path, max_angle_deviation_deg=15.0)
            if len(path) < 2:
                continue

            # Convert pixel coords to geographic CRS coordinates
            geo_coords = []
            widths = []
            probs = []

            for r, c in path:
                gx, gy = geo_transform_fn(float(c), float(r))
                geo_coords.append((gx, gy))
                widths.append(float(dist_transform[r, c]) * 2.0 * px_m)
                probs.append(float(road_prob_mask[r, c]))

            line = LineString(geo_coords)
            # Smooth out micro-pixel zig-zags
            line = line.simplify(0.000015, preserve_topology=True)
            if line.is_empty or len(line.coords) < 2:
                continue

            length_m = round(float(self._calculate_path_metric_length(path, px_m)), 2)
            avg_width_m = round(float(np.mean(widths)), 2)
            mean_conf = round(float(np.mean(probs)), 3)

            # Minimum continuity threshold
            min_len = max(25.0, self.min_road_length_meters) if px_m >= 0.3 else self.min_road_length_meters
            if length_m < min_len:
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

    def _calculate_path_metric_length(self, path: List[Tuple[int, int]], pixel_size_meters: Optional[float] = None) -> float:
        """Calculates metric length in meters along a polyline of pixel coordinates."""
        px_m = pixel_size_meters if pixel_size_meters is not None else self.pixel_size_meters
        if len(path) < 2:
            return 0.0
        total_m = 0.0
        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            dist_px = math.hypot(c1 - c0, r1 - r0)
            total_m += dist_px * px_m
        return total_m

    def _prune_spurs_metric(
        self,
        graph: nx.Graph,
        min_spur_meters: float = 8.0,
        pixel_size_meters: Optional[float] = None
    ) -> nx.Graph:
        """
        Recursively prunes dead-end degree-1 spurs shorter than min_spur_meters.
        Eliminates noise artifacts from parked vehicles, vegetation, and shadows.
        """
        px_m = pixel_size_meters if pixel_size_meters is not None else self.pixel_size_meters
        pruned = graph.copy()
        while True:
            endpoints = [n for n, deg in pruned.degree() if deg == 1]
            removed = False
            for ep in endpoints:
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

                metric_len = self._calculate_path_metric_length(path, px_m)
                if metric_len < min_spur_meters:
                    pruned.remove_nodes_from(path[:-1])
                    removed = True
            if not removed:
                break
        return pruned

    def _collapse_collinear_edges(
        self,
        path: List[Tuple[int, int]],
        max_angle_deviation_deg: float = 15.0
    ) -> List[Tuple[int, int]]:
        """
        Simplifies degree-2 intermediate nodes where incident vectors deviate by < 15°.
        Merges redundant vertices into a smooth, production-ready continuous polyline.
        """
        if len(path) <= 2:
            return path

        threshold_cos = math.cos(math.radians(max_angle_deviation_deg))
        simplified = [path[0]]

        for i in range(1, len(path) - 1):
            p_prev = simplified[-1]
            p_curr = path[i]
            p_next = path[i + 1]

            # Vector 1: p_prev -> p_curr
            v1_x = p_curr[1] - p_prev[1]
            v1_y = p_curr[0] - p_prev[0]
            len1 = math.hypot(v1_x, v1_y)

            # Vector 2: p_curr -> p_next
            v2_x = p_next[1] - p_curr[1]
            v2_y = p_next[0] - p_curr[0]
            len2 = math.hypot(v2_x, v2_y)

            if len1 < 1e-4 or len2 < 1e-4:
                continue

            # Dot product to check angle
            cos_theta = ((v1_x * v2_x) + (v1_y * v2_y)) / (len1 * len2)
            cos_theta = max(-1.0, min(1.0, cos_theta))

            # If direction changes significantly (> 15 deg), preserve node as an elbow/curve
            if cos_theta < threshold_cos:
                simplified.append(p_curr)

        simplified.append(path[-1])
        return simplified

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
        padded = np.pad(im, 1, mode='constant', constant_values=0)

        P2 = padded[0:h, 1:w+1]
        P3 = padded[0:h, 2:w+2]
        P4 = padded[1:h+1, 2:w+2]
        P5 = padded[2:h+2, 2:w+2]
        P6 = padded[2:h+2, 1:w+1]
        P7 = padded[2:h+2, 0:w]
        P8 = padded[1:h+1, 0:w]
        P9 = padded[0:h, 0:w]

        # Condition 1: 2 <= B(P1) <= 6
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        c1 = (B >= 2) & (B <= 6)

        # Condition 2: A(P1) == 1
        A = (
            ((P2 == 0) & (P3 == 1)).astype(int) +
            ((P3 == 0) & (P4 == 1)).astype(int) +
            ((P4 == 0) & (P5 == 1)).astype(int) +
            ((P5 == 0) & (P6 == 1)).astype(int) +
            ((P6 == 0) & (P7 == 1)).astype(int) +
            ((P7 == 0) & (P8 == 1)).astype(int) +
            ((P8 == 0) & (P9 == 1)).astype(int) +
            ((P9 == 0) & (P2 == 1)).astype(int)
        )
        c2 = (A == 1)

        if iter_num == 0:
            c3 = (P2 * P4 * P6 == 0)
            c4 = (P4 * P6 * P8 == 0)
        else:
            c3 = (P2 * P4 * P8 == 0)
            c4 = (P2 * P6 * P8 == 0)

        marker = (im == 1) & c1 & c2 & c3 & c4
        return marker.astype(np.uint8)

    def _build_graph_from_skeleton(self, skeleton: np.ndarray) -> nx.Graph:
        """Constructs an 8-connected spatial Graph from skeleton pixels."""
        graph = nx.Graph()
        points = np.argwhere(skeleton == 255)

        for r, c in points:
            graph.add_node((int(r), int(c)))

        point_set = set((int(r), int(c)) for r, c in points)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        for r, c in points:
            node = (int(r), int(c))
            for dr, dc in offsets:
                nbr = (r + dr, c + dc)
                if nbr in point_set:
                    graph.add_edge(node, nbr)

        return graph

    def _extract_edge_paths(self, graph: nx.Graph) -> List[List[Tuple[int, int]]]:
        """Compresses chains of degree-2 nodes into individual LineString paths."""
        paths = []
        visited_edges = set()

        junctions = [n for n, deg in graph.degree() if deg != 2]
        if not junctions and graph.number_of_nodes() > 0:
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
