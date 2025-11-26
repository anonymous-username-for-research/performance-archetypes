import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class RQ2:
    """
    RQ2: Performance Archetypes
    """

    def __init__(self, program_names: List[str], base_path: str = ".", output_dir: str = "rq2_results"):
        """
        Initialize RQ2 analysis.

        Args:
            program_names: List of program names to analyze
            base_path: Base directory containing data
            output_dir: Directory to save results
        """
        self.program_names = program_names
        self.base_path = Path(base_path)
        self.critical_paths_base = self.base_path / "critical-paths"
        self.static_analysis_base = self.base_path / "statistical-analysis"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.path_features = []  # List of all path feature vectors
        self.path_metadata = []  # Metadata for each path
        self.archetypes = None    # Discovered archetypes
        self.optimal_k = None     # Optimal number of clusters
        self.max_path_depth = None  # Maximum path depth for dynamic thresholds
        self.resource_thresholds = None  # Resource usage thresholds for classification

        self.static_data = {}

        # Results storage
        self.results = {
            'archetype_definitions': {},
            'archetype_distributions': {},
            'resource_phases': {},
            'cross_app_patterns': {},
            'stability_metrics': {},
            'archetype_examples': {}
        }

        print(f"RQ2 Analysis initialized for programs: {', '.join(program_names)}")

    def run_full_analysis(self):
        """Execute complete RQ2 analysis pipeline."""
        print("\n" + "="*80)
        print("Starting RQ2: Performance Archetypes Analysis")
        print("="*80)

        # Step 1: Load static analysis for function characterization
        print("\n[Step 1/8] Loading static analysis data...")
        self.load_static_analysis()

        # Step 2: Extract path features from all executions
        print("\n[Step 2/8] Extracting path features...")
        self.extract_all_path_features()

        # Step 3: Discover optimal number of archetypes
        print("\n[Step 3/8] Determining optimal number of archetypes...")
        self.determine_optimal_clusters()

        # Step 4: Perform archetype clustering
        print("\n[Step 4/8] Discovering performance archetypes...")
        self.discover_archetypes()

        # Step 5: Characterize each archetype
        print("\n[Step 5/8] Characterizing archetypes...")
        self.characterize_archetypes()

        # Step 6: Analyze archetype stability
        print("\n[Step 6/8] Analyzing archetype stability...")
        self.analyze_archetype_stability()

        # Step 7: Find cross-application patterns
        print("\n[Step 7/8] Finding cross-application patterns...")
        self.find_cross_application_patterns()

        # Step 8: Generate outputs
        print("\n[Step 8/8] Generating visualizations and reports...")
        self.generate_performance_signatures()
        self.save_results()

        print("\n" + "="*80)
        print("RQ2 Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

    def load_static_analysis(self):
        """Load static analysis data for all programs."""
        for program in self.program_names:
            static_path = self.static_analysis_base / program / "analysis" / "statistical_function_analysis.json"

            if not static_path.exists():
                print(f"  Warning: Static analysis not found for {program}")
                self.static_data[program] = {}
                continue

            with open(static_path, 'r') as f:
                data = json.load(f)

            functions = {}
            for file_data in data:
                for func in file_data.get('functions', []):
                    func_name = func['name'].strip().lower()
                    functions[func_name] = func

            self.static_data[program] = functions
            print(f"  Loaded {len(functions)} functions for {program}")

    def timestamp_to_ns(self, timestamp_str: str) -> int:
        """Convert ISO timestamp string to nanoseconds."""
        from datetime import timezone
        if '.' in timestamp_str:
            main_part, nano_part = timestamp_str.rsplit('.', 1)
            dt = datetime.fromisoformat(main_part).replace(tzinfo=timezone.utc)
            ns = int(dt.timestamp() * 1e9) + int(nano_part)
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            ns = int(dt.timestamp() * 1e9)
        return ns

    def extract_all_path_features(self):
        """Extract features from all critical paths across all programs."""
        total_paths = 0

        for program in self.program_names:
            print(f"\n  Processing {program}...")
            program_path = self.critical_paths_base / program / "analysis"

            if not program_path.exists():
                print(f"    Program path not found: {program_path}")
                continue

            for input_num in range(500):
                input_dir = program_path / f"{program}_analysis_itself_input{input_num}"

                if not input_dir.exists():
                    continue

                for iter_num in range(3):
                    iter_dir = input_dir / f"iter-{iter_num}"
                    if program != "sqlite":
                        critical_path_file = input_dir / f"iter-{iter_num}" / "critical_paths.json"
                    else:
                        critical_path_file = input_dir / f"iter-{iter_num}" / "critical_paths_with_functions.json"

                    kernel_data_file = iter_dir / "kernel_data.json"

                    if not (critical_path_file.exists() and kernel_data_file.exists()):
                        continue

                    with open(critical_path_file, 'r') as f:
                        critical_data = json.load(f)
                    with open(kernel_data_file, 'r') as f:
                        kernel_data = json.load(f)

                    if 'top_k_critical_paths' not in critical_data:
                        continue

                    for path_idx, path in enumerate(critical_data['top_k_critical_paths']):
                        features = self.extract_path_features(
                            path, kernel_data, program, input_num, iter_num, path_idx
                        )

                        if features is not None:
                            self.path_features.append(features['vector'])
                            self.path_metadata.append(features['metadata'])
                            total_paths += 1

            print(f"    Extracted features from {total_paths} paths")

        print(f"\n  Total paths analyzed: {total_paths}")

        # Calculate statistics for dynamic thresholds
        if self.path_features:
            path_depths = [features[0] for features in self.path_features]
            self.max_path_depth = max(path_depths)

            cpu_means = [features[6] for features in self.path_features]
            memory_growths = [abs(features[9]) for features in self.path_features]
            io_intensities = [features[12] for features in self.path_features]

            self.resource_thresholds = {
                'cpu_high': np.percentile(cpu_means, 80),
                'cpu_medium': np.percentile(cpu_means, 40),
                'memory_high': np.percentile(memory_growths, 80),
                'memory_medium': np.percentile(memory_growths, 40),
                'io_high': np.percentile(io_intensities, 80),
                'io_medium': np.percentile(io_intensities, 40),
            }

    def extract_path_features(self, path: Dict, kernel_data: Dict,
                              program: str, input_num: int, iter_num: int,
                              path_idx: int) -> Optional[Dict]:
        """Extract comprehensive features from a single critical path."""
        try:
            functions = path['functions']
            if not functions:
                return None

            path_start_ns = self.timestamp_to_ns(functions[0]['start_time'])
            path_end_ns = self.timestamp_to_ns(functions[-1]['end_time'])
            path_duration_ns = path_end_ns - path_start_ns

            # 1. Structural features
            structural = self.extract_structural_features(path, program)

            # 2. Temporal features
            temporal = self.extract_temporal_features(path)

            # 3. Resource features
            resource = self.extract_resource_features(
                path_start_ns, path_end_ns, kernel_data
            )

            # 4. Transition features
            transition = self.extract_transition_features(path, program)

            # 5. Phase features
            phase = self.extract_phase_features(path, kernel_data)

            # Combine all features into a single vector
            feature_vector = np.concatenate([
                structural, temporal, resource, transition, phase
            ])

            metadata = {
                'program': program,
                'input': input_num,
                'iteration': iter_num,
                'path_rank': path['rank'],
                'path_idx': path_idx,
                'duration_ns': path_duration_ns,
                'path_length': len(functions),
                'thread_id': path.get('thread_id', 'unknown')
            }

            return {
                'vector': feature_vector,
                'metadata': metadata
            }

        except Exception as e:
            print(f"    Error extracting features: {e}")
            return None

    def extract_structural_features(self, path: Dict, program: str) -> np.ndarray:
        """Extract structural features from path."""
        functions = path['functions']

        path_depth = len(functions)
        complexities = []

        for func in functions:
            func_name = func['function_name'].strip().lower()
            if program in self.static_data and func_name in self.static_data[program]:
                static_info = self.static_data[program][func_name]
                complexity = (
                    static_info.get('line_of_codes', 0) * 0.3 +
                    static_info.get('number_of_loops', 0) * 0.3 +
                    static_info.get('number_of_nested_loops', 0) * 0.2 +
                    static_info.get('number_of_calls', 0) * 0.2
                )
                complexities.append(complexity)
            else:
                complexities.append(0)

        if complexities:
            complexity_mean = np.mean(complexities)
            complexity_std = np.std(complexities)
            complexity_gradient = np.polyfit(range(len(complexities)), complexities, 1)[0] if len(complexities) > 1 else 0
        else:
            complexity_mean = complexity_std = complexity_gradient = 0

        unique_prefixes = len(set(f['function_name'].split('_')[0] if '_' in f['function_name'] else f['function_name'][:5]
                                  for f in functions))

        return np.array([
            path_depth,
            np.log1p(path_depth),
            unique_prefixes,
            complexity_mean,
            complexity_std,
            complexity_gradient
        ])

    def extract_temporal_features(self, path: Dict) -> np.ndarray:
        """Extract temporal distribution features."""
        functions = path['functions']

        durations = [f['duration_ns'] for f in functions]
        self_times = [f['self_time_ns'] for f in functions]

        total_duration = sum(durations)

        def gini_coefficient(values):
            if len(values) < 2 or sum(values) == 0:
                return 0
            sorted_vals = sorted(values)
            n = len(values)
            cumsum = np.cumsum(sorted_vals)
            return (2 * np.sum((np.arange(1, n+1)) * sorted_vals)) / (n * cumsum[-1]) - (n + 1) / n

        time_gini = gini_coefficient(durations)

        # Bottleneck analysis
        if total_duration > 0:
            max_duration_ratio = max(durations) / total_duration
            top3_duration_ratio = sum(sorted(durations, reverse=True)[:3]) / total_duration if len(durations) >= 3 else max_duration_ratio
        else:
            max_duration_ratio = top3_duration_ratio = 0

        # Self-time ratio (computation vs waiting)
        total_self_time = sum(self_times)
        self_time_ratio = total_self_time / total_duration if total_duration > 0 else 0

        # Position of peak
        if durations:
            peak_position = durations.index(max(durations)) / len(durations)
        else:
            peak_position = 0

        return np.array([
            np.log1p(total_duration),
            time_gini,
            max_duration_ratio,
            top3_duration_ratio,
            self_time_ratio,
            peak_position
        ])

    def extract_resource_features(self, start_ns: int, end_ns: int,
                                  kernel_data: Dict) -> np.ndarray:
        """Extract resource consumption features for the path duration."""
        time_series = kernel_data.get('time_series', {})

        if not time_series or 'timestamps_ns' not in time_series:
            return np.zeros(9)  # Return zeros if no kernel data

        timestamps = np.array(time_series['timestamps_ns'])
        cpu_usage = np.array(time_series.get('cpu_usage', []))
        memory_usage = np.array(time_series.get('memory_usage', []))
        disk_usage = np.array(time_series.get('disk_usage', []))

        # Find indices within path duration
        mask = (timestamps >= start_ns) & (timestamps <= end_ns)

        if not np.any(mask):
            if not hasattr(self, '_timestamp_mismatch_warned'):
                self._timestamp_mismatch_warned = True
            cpu_subset = cpu_usage
            memory_subset = memory_usage
            disk_subset = disk_usage
        else:
            cpu_subset = cpu_usage[mask] if len(cpu_usage) > 0 else []
            memory_subset = memory_usage[mask] if len(memory_usage) > 0 else []
            disk_subset = disk_usage[mask] if len(disk_usage) > 0 else []

        # CPU features
        cpu_mean = np.mean(cpu_subset) if len(cpu_subset) > 0 else 0
        cpu_std = np.std(cpu_subset) if len(cpu_subset) > 0 else 0
        cpu_max = np.max(cpu_subset) if len(cpu_subset) > 0 else 0

        # Memory features
        if len(memory_subset) > 0:
            memory_growth = (memory_subset[-1] - memory_subset[0]) / 1e6  # Convert to MB
            memory_mean = np.mean(memory_subset) / 1e6  # MB
            memory_volatility = np.std(np.diff(memory_subset)) / 1e6 if len(memory_subset) > 1 else 0
        else:
            memory_growth = memory_mean = memory_volatility = 0

        # Disk I/O features
        if len(disk_subset) > 0:
            io_intensity = np.mean(disk_subset) / 1e6  # MB
            io_burstiness = np.std(disk_subset) / 1e6 if np.mean(disk_subset) > 0 else 0
            io_peak = np.max(disk_subset) / 1e6
        else:
            io_intensity = io_burstiness = io_peak = 0

        return np.array([
            cpu_mean, cpu_std, cpu_max,
            memory_growth, memory_mean, memory_volatility,
            io_intensity, io_burstiness, io_peak
        ])

    def extract_transition_features(self, path: Dict, program: str) -> np.ndarray:
        """Extract function type transition features."""
        functions = path['functions']

        # Get function types
        func_types = []
        for func in functions:
            func_name = func['function_name'].strip().lower()
            if program in self.static_data and func_name in self.static_data[program]:
                func_type = self.static_data[program][func_name].get('function_type', 'Unknown')
            else:
                func_type = 'Unknown'
            func_types.append(func_type)

        # Count transitions
        transition_counts = defaultdict(int)
        for i in range(len(func_types) - 1):
            transition = f"{func_types[i]}->{func_types[i+1]}"
            transition_counts[transition] += 1

        # Calculate transition entropy
        if transition_counts:
            total_transitions = sum(transition_counts.values())
            probs = [count/total_transitions for count in transition_counts.values()]
            transition_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            transition_entropy = 0

        # Type distribution
        type_distribution = Counter(func_types)
        cpu_ratio = type_distribution.get('CPU-bound', 0) / len(func_types) if func_types else 0
        memory_ratio = type_distribution.get('Memory-bound', 0) / len(func_types) if func_types else 0
        io_ratio = type_distribution.get('I/O-bound', 0) / len(func_types) if func_types else 0

        # Transition patterns
        io_to_cpu = transition_counts.get('I/O-bound->CPU-bound', 0)
        cpu_to_mem = transition_counts.get('CPU-bound->Memory-bound', 0)
        mem_to_io = transition_counts.get('Memory-bound->I/O-bound', 0)

        return np.array([
            transition_entropy,
            cpu_ratio, memory_ratio, io_ratio,
            io_to_cpu, cpu_to_mem, mem_to_io
        ])

    def extract_phase_features(self, path: Dict, kernel_data: Dict) -> np.ndarray:
        """Extract resource phase transition features."""
        functions = path['functions']

        if not functions:
            return np.zeros(5)

        # Divide path into phases
        phase_boundaries = [0, len(functions)//4, len(functions)//2, 3*len(functions)//4, len(functions)]

        phase_resources = []
        for i in range(4):
            start_idx = phase_boundaries[i]
            end_idx = phase_boundaries[i+1]

            if end_idx > start_idx:
                phase_funcs = functions[start_idx:end_idx]
                phase_start_ns = self.timestamp_to_ns(phase_funcs[0]['start_time'])
                phase_end_ns = self.timestamp_to_ns(phase_funcs[-1]['end_time'])

                phase_stats = self.extract_resource_features(phase_start_ns, phase_end_ns, kernel_data)
                phase_resources.append(phase_stats)

        if len(phase_resources) >= 2:
            # Calculate phase transition characteristics
            phase_array = np.array(phase_resources)

            # CPU phase transitions (count significant changes)
            cpu_phases = np.sum(np.abs(np.diff(phase_array[:, 0])) > 20)  # >20% CPU change

            # Memory phase pattern (monotonic growth?)
            memory_monotonic = 1 if all(np.diff(phase_array[:, 4]) >= 0) else 0

            # I/O concentration (which phase has most I/O?)
            io_by_phase = phase_array[:, 6]
            io_concentrated = 1 if np.max(io_by_phase) > 2 * np.mean(io_by_phase) else 0

            # Resource volatility across phases
            resource_volatility = np.mean(np.std(phase_array, axis=0))

            # Phase similarity
            if len(phase_resources) > 1:
                phase_distances = cdist(phase_array, phase_array, metric='euclidean')
                phase_similarity = np.mean(phase_distances[np.triu_indices_from(phase_distances, k=1)])
            else:
                phase_similarity = 0
        else:
            cpu_phases = memory_monotonic = io_concentrated = resource_volatility = phase_similarity = 0

        return np.array([
            cpu_phases,
            memory_monotonic,
            io_concentrated,
            resource_volatility,
            phase_similarity
        ])

    def determine_optimal_clusters(self):
        """Determine optimal number of clusters using silhouette score."""
        if len(self.path_features) < 10:
            self.optimal_k = 2
            return

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.path_features)

        self.features_normalized = features_normalized
        self.scaler = scaler

        k_range = range(2, min(15, len(self.path_features) // 5))
        silhouette_scores = []
        inertias = []

        print("  Testing different numbers of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # type: ignore
            labels = kmeans.fit_predict(features_normalized)

            silhouette = silhouette_score(features_normalized, labels)
            inertia = kmeans.inertia_

            silhouette_scores.append(silhouette)
            inertias.append(inertia)

            print(f"    k={k}: silhouette={silhouette:.3f}")

        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_k = list(k_range)[optimal_idx]

        print(f"\n  Optimal number of archetypes: {self.optimal_k}")
        print(f"    Best silhouette score: {silhouette_scores[optimal_idx]:.3f}")

    def discover_archetypes(self):
        """Perform clustering to discover performance archetypes."""
        print(f"\n  Clustering with k={self.optimal_k}...")

        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=20)  # type: ignore
        self.archetype_labels = self.kmeans.fit_predict(self.features_normalized)

        silhouette = silhouette_score(self.features_normalized, self.archetype_labels)
        calinski = calinski_harabasz_score(self.features_normalized, self.archetype_labels)
        davies_bouldin = davies_bouldin_score(self.features_normalized, self.archetype_labels)

        print(f"  Clustering complete!")
        print(f"    Silhouette score: {silhouette:.3f}")
        print(f"    Calinski-Harabasz index: {calinski:.1f}")
        print(f"    Davies-Bouldin index: {davies_bouldin:.3f}")

        self.results['clustering_metrics'] = {
            'optimal_k': self.optimal_k,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin
        }

    def characterize_archetypes(self):
        """Characterize each discovered archetype."""
        print("\n  Characterizing archetypes...")

        archetype_chars = {}

        for archetype_id in range(self.optimal_k or 0):
            archetype_mask = self.archetype_labels == archetype_id
            archetype_paths_normalized = self.features_normalized[archetype_mask]
            archetype_metadata = [self.path_metadata[i] for i, mask in enumerate(archetype_mask) if mask]

            # Get feature statistics (on normalized features)
            feature_means = np.mean(archetype_paths_normalized, axis=0)
            feature_stds = np.std(archetype_paths_normalized, axis=0)

            # Denormalize for interpretation
            feature_means_denorm = self.scaler.inverse_transform([feature_means])[0]

            interpretation = self.interpret_features(feature_means_denorm)

            app_distribution = Counter([m['program'] for m in archetype_metadata])

            avg_duration = np.mean([m['duration_ns'] for m in archetype_metadata])
            avg_length = np.mean([m['path_length'] for m in archetype_metadata])

            archetype_chars[f"Archetype_{archetype_id}"] = {
                'size': len(archetype_paths_normalized),
                'percentage': len(archetype_paths_normalized) / len(self.path_features) * 100,
                'interpretation': interpretation,
                'app_distribution': dict(app_distribution),
                'avg_duration_ms': abs(avg_duration / 1e6),
                'avg_path_length': avg_length,
                'feature_means': feature_means_denorm.tolist(),
                'feature_stds': feature_stds.tolist(),
                'representative_paths': self.find_representative_paths(archetype_id)
            }

        self.results['archetype_definitions'] = archetype_chars

    def interpret_features(self, features: np.ndarray) -> Dict:
        """Interpret feature vector into human-readable characteristics."""
        # Feature indices (based on extraction order)
        idx = 0

        # Structural features
        path_depth = features[idx]
        complexity_mean = features[idx+3]
        idx += 6

        # Temporal features
        duration = np.exp(features[idx]) - 1
        time_gini = features[idx+1]
        max_duration_ratio = features[idx+2]
        self_time_ratio = features[idx+4]
        idx += 6

        # Resource features
        cpu_mean = features[idx]
        memory_growth = features[idx+3]
        io_intensity = features[idx+6]
        idx += 9

        # Transition features
        transition_entropy = features[idx]
        cpu_ratio = features[idx+1]
        memory_ratio = features[idx+2]
        io_ratio = features[idx+3]
        idx += 7

        # Phase features
        cpu_phases = features[idx]
        memory_monotonic = features[idx+1]

        primary_chars = []

        # Depth classification
        if self.max_path_depth is not None:
            shallow_threshold = self.max_path_depth * 0.33
            medium_threshold = self.max_path_depth * 0.66

            if path_depth <= shallow_threshold:
                primary_chars.append("SHALLOW")
            elif path_depth <= medium_threshold:
                primary_chars.append("MEDIUM_DEPTH")
            else:
                primary_chars.append("DEEP")
        else:
            if path_depth < 5:
                primary_chars.append("SHALLOW")
            elif path_depth < 15:
                primary_chars.append("MEDIUM_DEPTH")
            else:
                primary_chars.append("DEEP")

        # Time distribution
        if time_gini > 0.7:
            primary_chars.append("CONCENTRATED")
        elif time_gini < 0.3:
            primary_chars.append("BALANCED")

        # Resource profile
        resource_profile = []

        if self.resource_thresholds is not None:
            if cpu_mean > self.resource_thresholds['cpu_high']:
                resource_profile.append("CPU_INTENSIVE")
            elif cpu_mean > self.resource_thresholds['cpu_medium']:
                resource_profile.append("CPU_MODERATE")

            if abs(memory_growth) > self.resource_thresholds['memory_high']:
                resource_profile.append("MEMORY_GROWING")
            elif abs(memory_growth) > self.resource_thresholds['memory_medium']:
                resource_profile.append("MEMORY_MODERATE")

            if io_intensity > self.resource_thresholds['io_high']:
                resource_profile.append("IO_HEAVY")
            elif io_intensity > self.resource_thresholds['io_medium']:
                resource_profile.append("IO_MODERATE")
        else:
            if cpu_mean > 50:
                resource_profile.append("CPU_INTENSIVE")
            if abs(memory_growth) > 10:
                resource_profile.append("MEMORY_GROWING")
            if io_intensity > 5:
                resource_profile.append("IO_HEAVY")

        if not resource_profile:
            resource_profile.append("LIGHTWEIGHT")

        if cpu_ratio > 0.5:
            dominant_type = "CPU_BOUND"
        elif memory_ratio > 0.5:
            dominant_type = "MEMORY_BOUND"
        elif io_ratio > 0.5:
            dominant_type = "IO_BOUND"
        else:
            dominant_type = "MIXED"

        return {
            'primary': ", ".join(primary_chars[:3]),
            'resource_profile': ", ".join(resource_profile),
            'dominant_type': dominant_type,
            'phase_behavior': "PHASED" if cpu_phases > 2 else "STABLE",
            'memory_pattern': "MONOTONIC_GROWTH" if memory_monotonic > 0.5 else "VOLATILE"
        }

    def find_representative_paths(self, archetype_id: int, n: int = 3) -> List[Dict]:
        """Find the most representative paths for an archetype (closest to centroid)."""
        archetype_mask = self.archetype_labels == archetype_id
        archetype_features = self.features_normalized[archetype_mask]
        archetype_indices = np.where(archetype_mask)[0]

        if len(archetype_features) == 0:
            return []

        centroid = self.kmeans.cluster_centers_[archetype_id]
        distances = np.linalg.norm(archetype_features - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n]

        representative = []
        for idx in closest_indices:
            global_idx = archetype_indices[idx]
            metadata = self.path_metadata[global_idx]
            representative.append({
                'program': metadata['program'],
                'input': metadata['input'],
                'iteration': metadata['iteration'],
                'path_rank': metadata['path_rank'],
                'distance_to_centroid': distances[idx]
            })

        return representative

    def analyze_archetype_stability(self):
        """Analyze stability of archetypes across iterations and inputs."""
        print("\n  Analyzing archetype stability...")

        # Group paths by program, input, and iteration
        path_groups = defaultdict(list)
        for i, metadata in enumerate(self.path_metadata):
            key = (metadata['program'], metadata['input'])
            path_groups[key].append({
                'iteration': metadata['iteration'],
                'archetype': self.archetype_labels[i],
                'path_rank': metadata['path_rank']
            })

        # Calculate stability metrics
        iteration_stability = []  # Same input, different iterations
        input_diversity = defaultdict(list)  # Different inputs, same program

        for (program, input_num), iterations in path_groups.items():
            if len(iterations) > 1:
                # Check if same archetype across iterations
                archetypes = [it['archetype'] for it in iterations if it['path_rank'] == 1]
                if len(archetypes) > 1:
                    stability = len(set(archetypes)) == 1
                    iteration_stability.append(stability)

            # Collect for input diversity analysis
            input_diversity[program].append(iterations[0]['archetype'] if iterations else -1)

        # Calculate metrics
        iteration_stability_score = np.mean(iteration_stability) if iteration_stability else 0

        program_diversity = {}
        for program, archetypes in input_diversity.items():
            unique_archetypes = len(set(archetypes))
            total_inputs = len(archetypes)
            diversity = unique_archetypes / (self.optimal_k or 1)
            program_diversity[program] = {
                'unique_archetypes': unique_archetypes,
                'total_inputs': total_inputs,
                'diversity_score': diversity
            }

        # Archetype purity
        archetype_purity = {}
        for archetype_id in range(self.optimal_k or 2):
            mask = self.archetype_labels == archetype_id
            if np.any(mask):
                features = self.features_normalized[mask]
                if len(features) > 1:
                    distances = cdist(features, features, metric='euclidean')
                    avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    all_distances = cdist(self.features_normalized[:100], self.features_normalized[:100], metric='euclidean')
                    overall_avg = np.mean(all_distances[np.triu_indices_from(all_distances, k=1)])
                    purity = 1 - (avg_distance / overall_avg) if overall_avg > 0 else 0
                else:
                    purity = 1.0
                archetype_purity[f"Archetype_{archetype_id}"] = purity

        self.results['stability_metrics'] = {
            'iteration_stability': iteration_stability_score,
            'program_diversity': program_diversity,
            'archetype_purity': archetype_purity
        }

    def find_cross_application_patterns(self):
        """Find patterns that appear across multiple applications."""
        print("\n  Finding cross-application patterns...")

        # Analyze which archetypes appear in multiple applications
        cross_app_archetypes = {}

        for archetype_id in range(self.optimal_k or 2):
            mask = self.archetype_labels == archetype_id
            metadata = [self.path_metadata[i] for i, m in enumerate(mask) if m]

            apps = set(m['program'] for m in metadata)

            if len(apps) > 1:
                app_counts = Counter(m['program'] for m in metadata)

                paths_by_app = defaultdict(list)
                for i, m in enumerate(mask):
                    if m:
                        paths_by_app[self.path_metadata[i]['program']].append(i)

                cross_app_archetypes[f"Archetype_{archetype_id}"] = {
                    'applications': sorted(apps),
                    'universality': len(apps) / len(self.program_names),
                    'distribution': dict(app_counts),
                    'total_instances': len(metadata)
                }

        # Find universal patterns (appear in all or most applications)
        universal_patterns = []
        for arch_name, arch_data in cross_app_archetypes.items():
            if arch_data['universality'] >= 0.5:  # Present in at least 50% of apps
                universal_patterns.append({
                    'archetype': arch_name,
                    'universality': arch_data['universality'],
                    'instances': arch_data['total_instances'],
                    'interpretation': self.results['archetype_definitions'][arch_name]['interpretation']
                })

        universal_patterns.sort(key=lambda x: x['archetype'])

        self.results['cross_app_patterns'] = {
            'cross_app_archetypes': cross_app_archetypes,
            'universal_patterns': universal_patterns
        }

    def generate_performance_signatures(self):
        """Generate application performance signatures"""
        if not hasattr(self, 'archetype_labels') or len(self.archetype_labels) == 0:
            print("  No archetype labels found. Skipping performance signatures.")
            return

        app_archetype_counts = defaultdict(lambda: defaultdict(int))

        for i, metadata in enumerate(self.path_metadata):
            program = metadata['program']
            archetype_id = self.archetype_labels[i]
            app_archetype_counts[program][archetype_id] += 1

        # Convert counts to percentages for each application
        app_percentages = {}
        for program in self.program_names:
            if program not in app_archetype_counts:
                continue

            total_paths = sum(app_archetype_counts[program].values())
            if total_paths == 0:
                continue

            percentages = {}
            for archetype_id in range(self.optimal_k or 2):
                count = app_archetype_counts[program].get(archetype_id, 0)
                percentages[archetype_id] = (count / total_paths) * 100

            app_percentages[program] = percentages

        if not app_percentages:
            print("  No application data available for signatures.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        fig.suptitle('Application Performance Signatures', fontsize=22, fontweight='bold')

        elegant_base_colors = [
            '#2E86AB',  # Sophisticated blue
            '#A23B72',  # Rich purple
            '#F18F01',  # Warm orange
            '#C73E1D',  # Deep red
            '#6A994E',  # Forest green
            '#BC4749',  # Muted red
            '#7209B7',  # Royal purple
            '#F77F00',  # Burnt orange
            '#06A77D',  # Teal
            '#118AB2',  # Ocean blue
            '#9B5DE5',  # Lavender
            '#F15BB5',  # Pink
            '#00F5FF',  # Cyan
            '#FF006E',  # Magenta
            '#8338EC',  # Violet
            '#3A86FF',  # Bright blue
            '#06FFA5',  # Mint
            '#FFBE0B',  # Gold
            '#FB5607',  # Coral
            '#FF006E'   # Rose
        ]

        colors_list = elegant_base_colors[:self.optimal_k]

        archetype_colors = {}
        for archetype_id in range(self.optimal_k or 2):
            archetype_colors[archetype_id] = colors_list[archetype_id % len(colors_list)]

        applications = sorted(app_percentages.keys())
        n_apps = len(applications)
        n_archetypes = self.optimal_k or 2

        bottoms = np.zeros(n_apps)

        bars = []
        bar_labels = []

        for archetype_id in range(n_archetypes):
            percentages = [app_percentages[app].get(archetype_id, 0) for app in applications]

            if any(p > 0 for p in percentages):
                bar = ax.bar(applications, percentages, bottom=bottoms,
                             label=f'A{archetype_id}',
                             color=archetype_colors[archetype_id],
                             alpha=0.95, edgecolor='white', linewidth=1.2)
                bars.append(bar)
                bar_labels.append(f'A{archetype_id}')

                bottoms = bottoms + np.array(percentages)

        program_name_map = {
            "openssl": "OpenSSL",
            "sqlite": "SQLite 3",
            "zstd": "Zstandard",
            "ffmpeg": "FFmpeg"
        }

        ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Application', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(applications)))
        ax.set_xticklabels([program_name_map[applications[i]] for i in range(len(applications))], fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticklabels([f'{tick:.1f}%' for tick in range(0, 101, 10)], fontsize=14)

        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        for i, app in enumerate(applications):
            current_bottom = 0
            for archetype_id in range(n_archetypes):
                percentage = app_percentages[app].get(archetype_id, 0)
                if percentage > 5:
                    label_y = current_bottom + percentage / 2
                    ax.text(i, label_y, f'A{archetype_id} ({percentage:.1f}%)',
                            ha='center', va='center', fontsize=15, fontweight='bold',
                            color='white')
                current_bottom += percentage

        legend_handles = []
        legend_labels = []
        for archetype_id in range(n_archetypes):
            if any(app_percentages[app].get(archetype_id, 0) > 0 for app in applications):
                from matplotlib.patches import Patch
                handle = Patch(facecolor=archetype_colors[archetype_id],
                               alpha=0.95, edgecolor='white', linewidth=1.2)
                legend_handles.append(handle)
                legend_labels.append(f'A{archetype_id}')

        ax.legend(legend_handles, legend_labels, loc='upper left',
                  bbox_to_anchor=(1.02, 1), fontsize=13, framealpha=0.95,
                  title='Archetype', title_fontsize=14,
                  edgecolor='#e0e0e0', frameon=True, fancybox=False, shadow=False)

        plt.tight_layout()
        output_path = self.output_dir / 'rq2_performance_signatures.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.show()

        print(f"  Performance signatures saved to {output_path}")

    def save_results(self):
        """Save all analysis results."""
        results_summary = {
            'results': self.results
        }

        json_path = self.output_dir / 'rq2_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(f"  Results saved to {json_path}")

        # Save markdown report
        self.save_markdown_report()

        # Save archetype assignments
        assignments_df = pd.DataFrame({
            'program': [m['program'] for m in self.path_metadata],
            'input': [m['input'] for m in self.path_metadata],
            'iteration': [m['iteration'] for m in self.path_metadata],
            'path_rank': [m['path_rank'] for m in self.path_metadata],
            'archetype': self.archetype_labels
        })
        csv_path = self.output_dir / 'archetype_assignments.csv'
        assignments_df.to_csv(csv_path, index=False)
        print(f"  Archetype assignments saved to {csv_path}")

    def save_markdown_report(self):
        """Generate and save markdown report."""
        report = []
        report.append("# RQ2: Performance Archetypes\n")
        report.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Programs Analyzed**: {', '.join(self.program_names)}\n")
        report.append(f"**Total Paths**: {len(self.path_features)}\n")
        report.append(f"**Discovered Archetypes**: {self.optimal_k}\n")

        # Clustering metrics
        if 'clustering_metrics' in self.results:
            metrics = self.results['clustering_metrics']
            report.append("\n## Clustering Quality\n")
            report.append(f"- Silhouette Score: {metrics['silhouette_score']:.3f}\n")
            report.append(f"- Calinski-Harabasz Index: {metrics['calinski_harabasz']:.1f}\n")
            report.append(f"- Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}\n")

        # Archetype definitions
        report.append("\n## Discovered Archetypes\n")
        for i in range(self.optimal_k or 2):
            if f"Archetype_{i}" in self.results['archetype_definitions']:
                arch = self.results['archetype_definitions'][f"Archetype_{i}"]
                report.append(f"\n### Archetype {i}\n")
                report.append(f"- **Size**: {arch['size']} paths ({arch['percentage']:.1f}%)\n")
                report.append(f"- **Characteristics**: {arch['interpretation']['primary']}\n")
                report.append(f"- **Resource Profile**: {arch['interpretation']['resource_profile']}\n")
                report.append(f"- **Dominant Type**: {arch['interpretation']['dominant_type']}\n")
                report.append(f"- **Average Duration**: {arch['avg_duration_ms']:.2f} ms\n")
                report.append(f"- **Average Path Length**: {arch['avg_path_length']:.1f} functions\n")

                if arch['app_distribution']:
                    report.append(f"- **Application Distribution**:\n")
                    for app, count in arch['app_distribution'].items():
                        report.append(f"  - {app}: {count} paths\n")

        # Stability analysis
        if 'stability_metrics' in self.results:
            report.append("\n## Stability Analysis\n")
            stability = self.results['stability_metrics']
            report.append(f"- **Iteration Stability**: {stability['iteration_stability']:.1%}\n")

            if 'program_diversity' in stability:
                report.append("\n### Program Diversity\n")
                for prog, div in stability['program_diversity'].items():
                    report.append(f"- **{prog}**: {div['unique_archetypes']} unique archetypes "
                                  f"(diversity: {div['diversity_score']:.2f})\n")

        # Cross-application patterns
        if 'cross_app_patterns' in self.results:
            patterns = self.results['cross_app_patterns']
            if patterns['universal_patterns']:
                report.append("\n## Universal Patterns\n")
                for pattern in patterns['universal_patterns']:
                    report.append(f"\n### {pattern['archetype']}\n")
                    report.append(f"- **Universality**: {pattern['universality']:.1%} of applications\n")
                    report.append(f"- **Instances**: {pattern['instances']}\n")
                    report.append(f"- **Characteristics**: {pattern['interpretation']['primary']}\n")

        report_path = self.output_dir / 'rq2_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report)

        print(f"  Markdown report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 Analysis: Performance Archetypes")
    parser.add_argument('programs', nargs='*', help='List of program names to analyze')
    args = parser.parse_args()

    programs = args.programs
    rq2 = RQ2(program_names=programs)
    rq2.run_full_analysis()
