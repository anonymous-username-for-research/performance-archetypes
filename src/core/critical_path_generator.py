
from tmll.tmll_client import TMLLClient
from tmll.common.models.experiment import Experiment
from tmll.ml.modules.custom.critical_path_module import CriticalPathAnalysisModule
from tmll.ml.modules.resource_optimization.idle_resource_detection_module import IdleResourceDetection


class CriticalPathGenerator:

    def __init__(self, tmll_client: TMLLClient, resample_freq: str = "1us", hotspots_top_n: int = 100):
        self.tmll_client = tmll_client
        self.resample_freq = resample_freq
        self.hotspots_top_n = hotspots_top_n

    def _extract_kernel_data(self, client: TMLLClient, experiment: Experiment) -> dict:
        outputs = experiment.find_outputs(keyword=["cpu usage", "ust memory usage", "disk i/o"],
                                          type="TREE_TIME_XY", match_any=True)

        module = IdleResourceDetection(client, experiment, resample_freq="100us", outputs=outputs)

        keys = ["CPU Usage", "Memory Usage", "Disk I/O View"]

        results = {}

        time_series = {
            "timestamps_ns": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": []
        }

        combined_df = module.data_preprocessor.combine_dataframes(list(module.dataframes.values()))

        time_series["timestamps_ns"] = combined_df.index.values.astype("int64").tolist()
        time_series["cpu_usage"] = combined_df["CPU Usage"].tolist()
        time_series["memory_usage"] = combined_df["Memory Usage"].tolist()
        time_series["disk_usage"] = combined_df["Disk I/O View"].tolist()

        results["time_series"] = time_series

        stats = {}
        for key in keys:
            series = combined_df[key].dropna()
            series = series[series > 0]
            if not series.empty: # type: ignore
                stats[key] = {
                    "peak": series.max(),
                    "average": series.mean(),
                    "median": series.median() # type: ignore
                }
            else:
                stats[key] = {
                    "peak": -1,
                    "average": -1,
                    "median": -1
                }

        results["aggregated_metrics"] = stats

        results["trace_start_ns"] = int(combined_df.index.min().value) # type: ignore
        results["trace_end_ns"] = int(combined_df.index.max().value) # type: ignore

        return results

    def get_critical_path(self, traces: list[dict[str, str]], experiment_name: str,
                          top_k_critical_paths: int = 1) -> tuple:
        experiment = self.tmll_client.create_experiment(traces=[traces[-1]],
                                                        experiment_name=f"{experiment_name}_uftrace")
        if not experiment:
            raise Exception("Experiment creation failed")

        flame_chart_output = experiment.find_outputs(keyword=["flame", "chart", "callstack"])
        if not flame_chart_output:
            raise Exception("No flame chart output found")

        cpa = CriticalPathAnalysisModule(client=self.tmll_client, experiment=experiment, resample_freq=self.resample_freq)

        critical_paths = cpa.get_critical_paths(k=top_k_critical_paths)
        function_stats = cpa.get_function_statistics()
        function_hotspots = cpa.get_hotspot_functions(top_n=self.hotspots_top_n)

        executed_functions = list(set(function_stats["function_name"].tolist()))
        critical_paths["executed_functions"] = executed_functions

        kernel_experiment = self.tmll_client.create_experiment(traces=traces[:-1],
                                                               experiment_name=f"{experiment_name}_lttng")
        if not kernel_experiment:
            raise Exception("Experiment creation failed")

        kernel_data = self._extract_kernel_data(self.tmll_client, kernel_experiment)

        return critical_paths, kernel_data, function_stats, function_hotspots
