import argparse
import fnmatch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')


class RQ1:
    """
    RQ1: The Static-Dynamic Performance Paradox
    """

    def __init__(self, program_names: List[str], base_path: str = ".", output_dir: str = "rq1_results"):
        """
        Initialize RQ1 analysis.

        Args:
            program_names: List of program names to analyze (e.g., ['sqlite', 'zstd', 'openssl'])
            base_path: Base directory containing 'critical-paths' and 'statistical-analysis' folders
            output_dir: Directory to save results
        """
        self.program_names = program_names
        self.base_path = Path(base_path)
        self.critical_paths_base = self.base_path / "critical-paths"
        self.static_analysis_base = self.base_path / "statistical-analysis"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data storage
        self.static_data = {}
        self.dynamic_data = {}

        # Analysis results
        self.results = {
            'static_complexity_scores': {},
            'dynamic_criticality_scores': {},
            'paradox_functions': {},
            'hidden_bottlenecks': {},
            'statistical_correlations': {}
        }

        print(f"RQ1 Analysis initialized for programs: {', '.join(program_names)}")

    def run_full_analysis(self):
        """Execute complete RQ1 analysis pipeline."""
        print("\n" + "="*80)
        print("Starting RQ1: The Static-Dynamic Paradox Analysis")
        print("="*80)

        # Step 1: Load all data
        print("\n[Step 1/7] Loading data...")
        self.load_all_data()

        # Step 3: Calculate dynamic criticality scores
        print("\n[Step 2/7] Computing dynamic criticality scores...")
        self.compute_dynamic_criticality_scores()

        # Step 3: Calculate static complexity scores
        print("\n[Step 3/7] Computing static complexity scores...")
        self.compute_static_complexity_scores()

        # Step 4: Identify paradox functions
        print("\n[Step 4/7] Identifying paradox functions...")
        self.identify_paradox_functions()

        # Step 5: Discover hidden bottlenecks
        print("\n[Step 5/7] Discovering hidden bottlenecks...")
        self.discover_hidden_bottlenecks()

        # Step 6: Calculate statistical correlations
        print("\n[Step 6/7] Calculating statistical correlations...")
        self.calculate_statistical_correlations()

        # Generate visualizations and reports
        print("\n[Final] Generating visualizations and reports...")
        self.generate_quadrant_plot()
        self.save_results()

        print("\n" + "="*80)
        print("RQ1 Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

    def load_all_data(self):
        """Load static analysis and dynamic execution data for all programs."""
        for program in self.program_names:
            print(f"  Loading data for {program}...")

            # Load static analysis
            self.static_data[program] = self.load_static_analysis(program)

            # Load dynamic execution data
            self.dynamic_data[program] = self.load_dynamic_data(program)

            # Modify some function names in dynamic data for consistency
            if program == "zstd":
                for _, input_data in self.dynamic_data[program].items():
                    for _, iter_data in input_data['iterations'].items():
                        if 'top_k_critical_paths' in iter_data:
                            for path in iter_data['top_k_critical_paths']:
                                for func in path['functions']:
                                    func_name = func['function_name'].strip()
                                    # If the function name ends with "_noDict_*", replace
                                    # the suffix with "_generic", so it would end with "_noDict_generic"
                                    if fnmatch.fnmatch(func_name, '*_noDict_*'):
                                        base_name = func_name.rsplit('_noDict_', 1)[0]
                                        func['function_name'] = f"{base_name}_noDict_generic"

    def load_static_analysis(self, program: str) -> Dict:
        """Load static analysis data for a program."""
        static_path = self.static_analysis_base / program / "analysis" / "statistical_function_analysis.json"

        if not static_path.exists():
            print(f"    Warning: Static analysis not found at {static_path}")
            return {}

        with open(static_path, 'r') as f:
            data = json.load(f)

        functions = {}
        for file_data in data:
            for func in file_data.get('functions', []):
                func_name = func['name'].strip()
                functions[func_name] = func
                func['static_complexity_score'] = self.calculate_static_complexity(func)

        return functions

    def calculate_static_complexity(self, func: Dict) -> float:
        """
        Calculate composite static complexity score for a function.

        Components:
        - Lines of code (LOC)
        - Cyclomatic complexity (branches + loops)
        - Nesting depth
        - Number of function calls
        - Memory operations
        """
        loc = func.get('line_of_codes', 0)
        loops = func.get('number_of_loops', 0)
        nested_loops = func.get('number_of_nested_loops', 0)
        branches = func.get('number_of_branches', {})
        calls = func.get('number_of_calls', 0)

        # Extract branch counts
        if_count = branches.get('number_of_if', 0)
        switch_count = branches.get('number_of_switch', 0)

        # Cyclomatic complexity approximation
        cyclomatic = if_count + switch_count + loops + 1

        # Nesting penalty
        nesting_penalty = 1 + (nested_loops * 0.5)

        # Weighted complexity score
        complexity = (
            0.2 * np.log1p(loc) +           # Log scale for LOC
            0.3 * np.log1p(cyclomatic) +    # Cyclomatic complexity
            0.2 * nesting_penalty +          # Nesting depth penalty
            0.2 * np.log1p(calls) +          # Function calls
            0.1 * (1 if func.get('has_io', False) else 0)  # I/O operations
        )

        return complexity

    def load_dynamic_data(self, program: str) -> Dict:
        """Load all dynamic execution traces for a program."""
        program_path = self.critical_paths_base / program / "analysis"

        if not program_path.exists():
            print(f"    Warning: Program path not found: {program_path}")
            return {}

        traces = {}
        missing_iterations = []

        for input_num in range(500):
            input_dir = program_path / f"{program}_analysis_itself_input{input_num}"

            if not input_dir.exists():
                continue

            traces[f"input_{input_num}"] = {'iterations': {}}

            for iter_num in range(3):
                if program != "sqlite":
                    iter_path = input_dir / f"iter-{iter_num}" / "critical_paths.json"
                else:
                    iter_path = input_dir / f"iter-{iter_num}" / "critical_paths_with_functions.json"

                if not iter_path.exists():
                    missing_iterations.append(f"input_{input_num}/iter-{iter_num}")
                    continue

                with open(iter_path, 'r') as f:
                    traces[f"input_{input_num}"]['iterations'][f"iter_{iter_num}"] = json.load(f)

        return traces

    def compute_static_complexity_scores(self):
        """Compute and rank functions by static complexity."""
        for program in self.program_names:
            if program not in self.static_data:
                continue

            func_scores = []
            for func_name, func_data in self.static_data[program].items():
                func_scores.append({
                    'function': func_name,
                    'complexity': func_data['static_complexity_score'],
                    'loc': func_data.get('line_of_codes', 0),
                    'loops': func_data.get('number_of_loops', 0),
                    'nested_loops': func_data.get('number_of_nested_loops', 0)
                })

            # Sort by complexity and assign ranks
            func_scores.sort(key=lambda x: (x['complexity'], x['function']), reverse=True)
            for rank, func in enumerate(func_scores, 1):
                func['static_rank'] = rank

            self.results['static_complexity_scores'][program] = func_scores

    def compute_dynamic_criticality_scores(self):
        """Compute dynamic criticality scores based on actual execution traces."""
        for program in self.program_names:
            if program not in self.dynamic_data:
                continue

            function_criticality = defaultdict(lambda: {
                'appearances': 0,
                'total_duration_ns': 0,
                'total_self_time_ns': 0,
                'rank_sum': 0,
                'path_positions': [],
                'bottleneck_count': 0
            })

            all_executed_functions = set()
            total_paths = 0

            # Process each input and iteration
            for _, input_data in self.dynamic_data[program].items():
                for _, iter_data in input_data['iterations'].items():
                    if 'executed_functions' in iter_data:
                        all_executed_functions.update(f.strip() for f in iter_data['executed_functions'])

                    if 'top_k_critical_paths' not in iter_data:
                        continue

                    for path in iter_data['top_k_critical_paths']:
                        total_paths += 1
                        path_rank = path['rank']

                        for pos, func in enumerate(path['functions']):
                            func_name = func['function_name'].strip()

                            function_criticality[func_name]['appearances'] += 1  # type: ignore
                            function_criticality[func_name]['total_duration_ns'] += func['duration_ns']
                            function_criticality[func_name]['total_self_time_ns'] += func['self_time_ns']
                            function_criticality[func_name]['rank_sum'] += path_rank
                            function_criticality[func_name]['path_positions'].append(pos / len(path['functions']))  # type: ignore

                        for bottleneck in path.get('bottlenecks', []):
                            func_name = bottleneck['function'].strip()
                            function_criticality[func_name]['bottleneck_count'] += 1  # type: ignore

            # Calculate criticality scores
            criticality_scores = []

            # First, process functions that appeared in critical paths
            for func_name, stats in function_criticality.items():
                if stats['appearances'] == 0:
                    continue

                avg_rank = stats['rank_sum'] / stats['appearances']  # type: ignore
                appearance_rate = stats['appearances'] / total_paths  # type: ignore
                avg_duration = stats['total_duration_ns'] / stats['appearances']  # type: ignore
                bottleneck_rate = stats['bottleneck_count'] / stats['appearances']  # type: ignore

                criticality = (
                    0.3 * (1 / avg_rank) +           # Lower rank = higher criticality
                    0.3 * appearance_rate +           # Frequency of appearance
                    0.2 * np.log1p(avg_duration) +   # Duration impact
                    0.2 * bottleneck_rate             # Bottleneck frequency
                )

                criticality_scores.append({
                    'function': func_name,
                    'criticality': criticality,
                    'appearances': stats['appearances'],
                    'avg_duration_ns': avg_duration,
                    'avg_self_time_ns': stats['total_self_time_ns'] / stats['appearances'],  # type: ignore
                    'appearance_rate': appearance_rate * 100,
                    'avg_rank': avg_rank,
                    'in_critical_paths': True
                })

            # Now add functions that were executed but never appeared in critical paths
            functions_in_critical_paths = set(function_criticality.keys())
            functions_not_in_paths = sorted(all_executed_functions - functions_in_critical_paths)

            max_avg_rank = max((f['avg_rank'] for f in criticality_scores), default=total_paths + 1)

            for idx, func_name in enumerate(functions_not_in_paths):
                # These functions have minimal criticality (executed but never critical)
                criticality_scores.append({
                    'function': func_name,
                    'criticality': 0.0,  # Not critical at all
                    'appearances': 0,
                    'avg_duration_ns': 0,
                    'avg_self_time_ns': 0,
                    'appearance_rate': 0.0,
                    'avg_rank': max_avg_rank + idx,  # Rank them at the bottom
                    'in_critical_paths': False
                })

            # Sort by criticality and assign ranks
            criticality_scores.sort(key=lambda x: (x['criticality'], x['function']), reverse=True)
            for rank, func in enumerate(criticality_scores, 1):
                func['dynamic_rank'] = rank

            self.results['dynamic_criticality_scores'][program] = criticality_scores

    def identify_paradox_functions(self,
                                   paradox_threshold=25,
                                   min_rank_difference=0.3):
        """
        Identify paradox functions using percentile-based approach.

        Args:
            paradox_threshold: Percentile threshold (e.g., 25 for top/bottom 25%)
            min_rank_difference: Minimum normalized rank difference to consider (0-1 scale)
        """

        for program in self.program_names:
            if program not in self.results['static_complexity_scores']:
                continue

            static_lookup = {f['function']: f for f in self.results['static_complexity_scores'][program]}
            dynamic_lookup = {f['function']: f for f in self.results['dynamic_criticality_scores'][program]}

            common_functions = sorted(set(static_lookup.keys()) & set(dynamic_lookup.keys()))

            if len(common_functions) < 10:
                print(f"  {program}: Insufficient data ({len(common_functions)} functions)")
                continue

            # Normalize ranks to [0, 1]
            static_ranks_norm = np.array([
                stats.percentileofscore(
                    [static_lookup[f]['static_rank'] for f in common_functions],
                    static_lookup[f]['static_rank'],
                    kind='rank'
                ) / 100
                for f in common_functions
            ])

            dynamic_ranks_norm = np.array([
                stats.percentileofscore(
                    [dynamic_lookup[f]['dynamic_rank'] for f in common_functions],
                    dynamic_lookup[f]['dynamic_rank'],
                    kind='rank'
                ) / 100
                for f in common_functions
            ])

            # Calculate rank differences
            rank_differences = static_ranks_norm - dynamic_ranks_norm

            # Calculate percentile thresholds
            static_high_threshold = np.percentile(static_ranks_norm, 100 - paradox_threshold)
            dynamic_low_threshold = np.percentile(dynamic_ranks_norm, paradox_threshold)
            dynamic_high_threshold = np.percentile(dynamic_ranks_norm, 100 - paradox_threshold)
            static_low_threshold = np.percentile(static_ranks_norm, paradox_threshold)

            # Calculate effect size (Cohen's d) for the rank difference
            mean_diff = np.mean(rank_differences)
            std_diff = np.std(rank_differences, ddof=1)

            paradox_functions = {
                'simple_but_critical': [],
                'complex_but_irrelevant': [],
                'matched': []
            }

            for idx, func_name in enumerate(common_functions):
                static_rank = static_lookup[func_name]['static_rank']
                dynamic_rank = dynamic_lookup[func_name]['dynamic_rank']

                static_norm = static_ranks_norm[idx]
                dynamic_norm = dynamic_ranks_norm[idx]
                rank_diff = rank_differences[idx]

                # Calculate standardized effect size for this function
                effect_size = (rank_diff - mean_diff) / std_diff if std_diff > 0 else 0

                # Determine category
                is_simple_critical = (
                    static_norm >= static_high_threshold and  # Simple (high static rank)
                    dynamic_norm <= dynamic_low_threshold and  # Critical (low dynamic rank)
                    rank_diff >= min_rank_difference  # Substantial difference
                )

                is_complex_irrelevant = (
                    static_norm <= static_low_threshold and  # Complex (low static rank)
                    dynamic_norm >= dynamic_high_threshold and  # Not critical (high dynamic rank)
                    rank_diff <= -min_rank_difference  # Substantial difference
                )

                func_info = {
                    'function': func_name,
                    'static_rank': static_rank,
                    'dynamic_rank': dynamic_rank,
                    'rank_difference': rank_diff,
                    'effect_size': effect_size,
                    'static_percentile': stats.percentileofscore(static_ranks_norm, static_norm),
                    'dynamic_percentile': stats.percentileofscore(dynamic_ranks_norm, dynamic_norm),
                    'static_complexity': static_lookup[func_name]['complexity'],
                    'dynamic_criticality': dynamic_lookup[func_name]['criticality'],
                    'loc': static_lookup[func_name].get('loc', 0),
                    'appearances': dynamic_lookup[func_name]['appearances']
                }

                if is_simple_critical:
                    paradox_functions['simple_but_critical'].append(func_info)
                elif is_complex_irrelevant:
                    paradox_functions['complex_but_irrelevant'].append(func_info)
                else:
                    paradox_functions['matched'].append(func_info)

            # Sort by effect size
            for category in paradox_functions:
                paradox_functions[category].sort(
                    key=lambda x: (abs(x['effect_size']), x['function']),
                    reverse=True
                )

            self.results['paradox_functions'][program] = paradox_functions

    def discover_hidden_bottlenecks(self):
        """Discover hidden bottlenecks - simple functions that dominate performance."""
        for program in self.program_names:
            if program not in self.results['dynamic_criticality_scores']:
                continue

            static_lookup = {f['function']: f for f in self.results.get('static_complexity_scores', {}).get(program, [])}

            hidden_bottlenecks = []

            # Analyze each critical function
            for func in self.results['dynamic_criticality_scores'][program][:20]:
                func_name = func['function']

                if func_name in static_lookup:
                    static_info = static_lookup[func_name]

                    total_functions = len(static_lookup)
                    is_simple = static_info['static_rank'] > total_functions / 2

                    if is_simple:
                        hidden_bottlenecks.append({
                            'function': func_name,
                            'dynamic_rank': func['dynamic_rank'],
                            'static_rank': static_info['static_rank'],
                            'criticality': func['criticality'],
                            'complexity': static_info['complexity'],
                            'avg_duration_ns': func['avg_duration_ns'],
                            'avg_self_time_ns': func['avg_self_time_ns'],
                            'appearance_rate': func['appearance_rate'],
                            'loc': static_info.get('loc', 0),
                            'impact_score': func['criticality'] / static_info['complexity']  # High = hidden bottleneck
                        })

            # Sort by impact score
            hidden_bottlenecks.sort(key=lambda x: (x['impact_score'], x['function']), reverse=True)

            self.results['hidden_bottlenecks'][program] = hidden_bottlenecks

    def test_normality(self, data: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test if data is normally distributed using multiple tests.

        Args:
            data: List of numeric values to test
            alpha: Significance level (default 0.05)

        Returns:
            Dictionary with test results and overall normality assessment
        """
        if len(data) < 3:
            return {
                'is_normal': False,
                'reason': 'insufficient_data',
                'shapiro_stat': None,
                'shapiro_p': None,
                'normaltest_stat': None,
                'normaltest_p': None
            }

        data_array = np.array(data)
        data_array = data_array[np.isfinite(data_array)]

        if len(data_array) < 3:
            return {
                'is_normal': False,
                'reason': 'insufficient_valid_data',
                'shapiro_stat': None,
                'shapiro_p': None,
                'normaltest_stat': None,
                'normaltest_p': None
            }

        results = {}

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(data_array)
        results['shapiro_stat'] = shapiro_stat
        results['shapiro_p'] = shapiro_p
        results['shapiro_normal'] = shapiro_p > alpha

        # D'Agostino-Pearson normality test
        if len(data_array) >= 8:
            normaltest_stat, normaltest_p = normaltest(data_array)
            results['normaltest_stat'] = normaltest_stat
            results['normaltest_p'] = normaltest_p
            results['normaltest_normal'] = normaltest_p > alpha
        else:
            results['normaltest_stat'] = None
            results['normaltest_p'] = None
            results['normaltest_normal'] = None

        # Overall assessment
        if results.get('shapiro_normal') is not None and results.get('normaltest_normal') is not None:
            results['is_normal'] = results['shapiro_normal'] and results['normaltest_normal']
        elif results.get('shapiro_normal') is not None:
            results['is_normal'] = results['shapiro_normal']
        elif results.get('normaltest_normal') is not None:
            results['is_normal'] = results['normaltest_normal']
        else:
            results['is_normal'] = None

        return results

    def calculate_statistical_correlations(self):
        """Calculate statistical correlations between static and dynamic complexity and criticality."""
        all_paradox_functions = []
        for program in self.program_names:
            if program in self.results['paradox_functions']:
                for category in ['simple_but_critical', 'complex_but_irrelevant']:
                    for func in self.results['paradox_functions'][program][category]:
                        func_copy = func.copy()
                        func_copy['program'] = program
                        func_copy['category'] = category
                        all_paradox_functions.append(func_copy)

        correlations = {}
        for program in self.program_names:
            if program not in self.results['paradox_functions']:
                continue

            # For Pearson: use actual continuous scores (complexity and criticality)
            static_scores = []
            dynamic_scores = []

            # For Spearman: use ranks (ordinal data)
            static_ranks = []
            dynamic_ranks = []

            static_lookup = {f['function']: f for f in self.results['static_complexity_scores'][program]}
            dynamic_lookup = {f['function']: f for f in self.results['dynamic_criticality_scores'][program]}

            for category in self.results['paradox_functions'][program]:
                for func in self.results['paradox_functions'][program][category]:
                    func_name = func['function']

                    # Collect ranks for Spearman
                    static_ranks.append(func['static_rank'])
                    dynamic_ranks.append(func['dynamic_rank'])

                    # Collect scores for Pearson
                    if func_name in static_lookup and func_name in dynamic_lookup:
                        static_scores.append(static_lookup[func_name]['complexity'])
                        dynamic_scores.append(dynamic_lookup[func_name]['criticality'])

            if len(static_ranks) > 2:
                # Spearman correlation on ranks
                spearman_corr, spearman_p = spearmanr(static_ranks, dynamic_ranks)

                # Pearson correlation on continuous scores
                pearson_corr = None
                pearson_p = None
                pearson_on_scores = False
                normality_tests = {}
                pearson_appropriate = False

                if len(static_scores) > 2 and len(static_scores) == len(dynamic_scores):
                    # Test normality of both variables
                    static_normality = self.test_normality(static_scores)
                    dynamic_normality = self.test_normality(dynamic_scores)

                    normality_tests = {
                        'static_normality': static_normality,
                        'dynamic_normality': dynamic_normality
                    }

                    both_normal = (
                        static_normality.get('is_normal') is True and
                        dynamic_normality.get('is_normal') is True
                    )
                    at_least_one_normal = (
                        static_normality.get('is_normal') is True or
                        dynamic_normality.get('is_normal') is True
                    )

                    pearson_corr, pearson_p = pearsonr(static_scores, dynamic_scores)
                    pearson_on_scores = True

                    pearson_appropriate = both_normal or at_least_one_normal
                else:
                    pearson_corr = None
                    pearson_p = None
                    pearson_on_scores = False
                    pearson_appropriate = False
                    normality_tests = {'note': 'insufficient_score_data_for_pearson'}

                correlations[program] = {
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'n_functions': len(static_ranks),
                    'pearson_on_scores': pearson_on_scores,
                    'pearson_appropriate': pearson_appropriate,
                    'normality_tests': normality_tests
                }

        self.results['statistical_correlations'] = correlations

    def generate_quadrant_plot(self):
        """Generate the Static-Dynamic Disconnect quadrant plot."""
        n_programs = len(self.program_names)
        if n_programs == 0:
            return

        n_cols = 2
        n_rows = (n_programs + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        fig.suptitle('The Static-Dynamic Paradox', fontsize=22, fontweight='bold')

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim == 1:
            axes = list(axes)
        else:
            axes = list(axes.flatten())

        for idx, program in enumerate(self.program_names):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get static and dynamic data for this program
            if program not in self.results['static_complexity_scores'] or \
               program not in self.results['dynamic_criticality_scores']:
                ax.text(0.5, 0.5, f'No data for {program}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(program.upper(), fontsize=12, fontweight='bold')
                continue

            static_lookup = {f['function']: f for f in self.results['static_complexity_scores'][program]}
            dynamic_lookup = {f['function']: f for f in self.results['dynamic_criticality_scores'][program]}

            static_only = set(static_lookup.keys()) - set(dynamic_lookup.keys())
            dynamic_only = set(dynamic_lookup.keys()) - set(static_lookup.keys())
            common_functions = sorted(set(static_lookup.keys()) & set(dynamic_lookup.keys()))

            if len(common_functions) == 0:
                ax.text(0.5, 0.5, f'No common functions for {program}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(program.upper(), fontsize=12, fontweight='bold')
                continue

            all_static_ranks = [static_lookup[f]['static_rank'] for f in common_functions]
            all_dynamic_ranks = [dynamic_lookup[f]['dynamic_rank'] for f in common_functions]

            min_static_rank = min(all_static_ranks) if all_static_ranks else 1
            max_static_rank = max(all_static_ranks) if all_static_ranks else 1
            min_dynamic_rank = min(all_dynamic_ranks) if all_dynamic_ranks else 1
            max_dynamic_rank = max(all_dynamic_ranks) if all_dynamic_ranks else 1

            static_norm = []
            dynamic_norm = []

            for func_name in common_functions:
                static_rank = static_lookup[func_name]['static_rank']
                dynamic_rank = dynamic_lookup[func_name]['dynamic_rank']

                if max_static_rank > min_static_rank:
                    static_norm_val = (max_static_rank - static_rank) / (max_static_rank - min_static_rank)
                else:
                    static_norm_val = 0.5

                if max_dynamic_rank > min_dynamic_rank:
                    dynamic_norm_val = (max_dynamic_rank - dynamic_rank) / (max_dynamic_rank - min_dynamic_rank)
                else:
                    dynamic_norm_val = 0.5

                static_norm.append(static_norm_val)
                dynamic_norm.append(dynamic_norm_val)

            func_category = {}
            if program in self.results['paradox_functions']:
                paradox_data = self.results['paradox_functions'][program]
                for func_info in paradox_data['simple_but_critical']:
                    func_category[func_info['function']] = 'simple_but_critical'
                for func_info in paradox_data['complex_but_irrelevant']:
                    func_category[func_info['function']] = 'complex_but_irrelevant'
                for func_info in paradox_data['matched']:
                    func_category[func_info['function']] = 'matched'

            simple_critical_x = []
            simple_critical_y = []
            complex_irrelevant_x = []
            complex_irrelevant_y = []
            matched_x = []
            matched_y = []

            for idx, func_name in enumerate(common_functions):
                category = func_category.get(func_name, 'matched')
                if category == 'simple_but_critical':
                    simple_critical_x.append(static_norm[idx])
                    simple_critical_y.append(dynamic_norm[idx])
                elif category == 'complex_but_irrelevant':
                    complex_irrelevant_x.append(static_norm[idx])
                    complex_irrelevant_y.append(dynamic_norm[idx])
                else:
                    matched_x.append(static_norm[idx])
                    matched_y.append(dynamic_norm[idx])

            if matched_x:
                ax.scatter(matched_x, matched_y, alpha=0.3, s=25, c='gray',
                           edgecolors='black', linewidth=0.3, label='Aligned', zorder=1)
            if simple_critical_x:
                ax.scatter(simple_critical_x, simple_critical_y, alpha=0.75, s=50,
                           c='red', edgecolors='darkred', linewidth=0.5,
                           label='Simple-but-Critical', zorder=3)
            if complex_irrelevant_x:
                ax.scatter(complex_irrelevant_x, complex_irrelevant_y, alpha=0.75, s=50,
                           c='blue', edgecolors='darkblue', linewidth=0.5,
                           label='Complex-but-Irrelevant', zorder=3)

            # Add quadrant lines
            ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add quadrant labels
            # Top-Left: Hidden Bottlenecks (Simple-but-Critical)
            ax.text(0.25, 0.75, 'Hidden Bottlenecks\n(Simple-but-Critical)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                    transform=ax.transAxes)

            # Bottom-Right: Misleading Complexity (Complex-but-Irrelevant)
            ax.text(0.75, 0.25, 'Misleading Complexity\n(Complex-but-Irrelevant)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    transform=ax.transAxes)

            # Top-Right: Aligned Behavior (Complex-and-Critical)
            ax.text(0.75, 0.75, 'Aligned Behavior',
                    ha='center', va='center', fontsize=11, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    transform=ax.transAxes)

            # Bottom-Left: Aligned Behavior (Simple-and-Non-Critical)
            ax.text(0.25, 0.25, 'Aligned Behavior',
                    ha='center', va='center', fontsize=11, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    transform=ax.transAxes)

            program_name_map = {
                "openssl": "OpenSSL",
                "sqlite": "SQLite 3",
                "zstd": "Zstandard",
                "ffmpeg": "FFmpeg"
            }

            # Set labels and title
            ax.set_xlabel('Static Complexity Rank\n(0 = Least Complex, 1 = Most Complex)', fontsize=16)
            ax.set_ylabel('Dynamic Criticality Rank\n(0 = Least Critical, 1 = Most Critical)', fontsize=16)
            ax.set_title(program_name_map[program], fontsize=18, fontweight='bold')
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontsize=14)
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=14)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            if func_category:
                ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        for idx in range(n_programs, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / 'rq1_quadrant_plot.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.show()

        print(f"  Quadrant plot saved to {output_path}")

    def save_results(self):
        """Save all analysis results to JSON file."""
        results_to_save = {
            'results': {
                'paradox_functions': {
                    program: {
                        'simple_but_critical': len(self.results['paradox_functions'][program]['simple_but_critical']),
                        'complex_but_irrelevant': len(self.results['paradox_functions'][program]['complex_but_irrelevant']),
                        'matched': len(self.results['paradox_functions'][program]['matched']),
                        'total_functions': (len(self.results['paradox_functions'][program]['simple_but_critical']) +
                                            len(self.results['paradox_functions'][program]['complex_but_irrelevant']) +
                                            len(self.results['paradox_functions'][program]['matched']))
                    }
                    for program in self.program_names
                    if program in self.results['paradox_functions']
                },
                'hidden_bottlenecks': self.results['hidden_bottlenecks'],
                'statistical_correlations': self.results['statistical_correlations']
            }
        }

        output_path = self.output_dir / 'rq1_results.json'
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        print(f"  Results saved to {output_path}")

        self.save_markdown_report()

    def save_markdown_report(self):
        """Generate and save a markdown report of the analysis."""
        report = []
        report.append("# RQ1: The Static-Dynamic Paradox\n")
        report.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Programs Analyzed**: {', '.join(self.program_names)}\n")

        report.append("\n## Detailed Results\n")

        for program in self.program_names:
            report.append(f"\n### {program.upper()}\n")

            # Paradox Functions
            if program in self.results['paradox_functions']:
                paradox = self.results['paradox_functions'][program]
                report.append(f"\n#### Paradox Analysis\n")
                report.append(f"- **Simple but Critical**: {len(paradox['simple_but_critical'])} functions\n")
                report.append(f"- **Complex but Irrelevant**: {len(paradox['complex_but_irrelevant'])} functions\n")
                report.append(f"- **Matched Expectations**: {len(paradox['matched'])} functions\n")

                # Top examples
                if paradox['simple_but_critical']:
                    report.append(f"\n**Top Simple but Critical Functions:**\n")
                    for func in paradox['simple_but_critical'][:5]:
                        report.append(f"- `{func['function']}`: static rank #{func['static_rank']} → "
                                      f"dynamic rank #{func['dynamic_rank']}\n")

            # Hidden Bottlenecks
            if program in self.results['hidden_bottlenecks'] and self.results['hidden_bottlenecks'][program]:
                report.append(f"\n#### Hidden Bottlenecks\n")
                report.append(f"Found {len(self.results['hidden_bottlenecks'][program])} hidden bottlenecks:\n")
                for hb in self.results['hidden_bottlenecks'][program][:5]:
                    report.append(f"- `{hb['function']}`: {hb['loc']} LOC, "
                                  f"impact score: {hb['impact_score']:.2f}\n")

            # Correlation
            if program in self.results['statistical_correlations']:
                corr = self.results['statistical_correlations'][program]
                report.append(f"\n#### Static-Dynamic Correlation\n")
                report.append(f"- Pearson r = {corr['pearson_correlation']:.3f} "
                              f"(p = {corr['pearson_p_value']:.4f})\n")
                report.append(f"- Spearman ρ = {corr['spearman_correlation']:.3f} "
                              f"(p = {corr['spearman_p_value']:.4f})\n")

        report_path = self.output_dir / 'rq1_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report)

        print(f"  Markdown report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ1 Analysis: The Static-Dynamic Paradox")
    parser.add_argument('programs', nargs='*', help='List of program names to analyze')
    args = parser.parse_args()

    programs = args.programs

    rq1 = RQ1(program_names=programs)
    rq1.run_full_analysis()
