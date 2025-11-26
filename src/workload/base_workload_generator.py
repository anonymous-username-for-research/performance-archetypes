import json
import logging
import math
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import gzip
import zipfile

from .utils.metrics_collerctor import MetricsCollector
from .utils.resource_manager import ResourceManager
from .regression.regression_manager import RegressionManager

from src.services.db_manager import DBManager
from src.services.uftrace_service import UftraceService

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WorkloadGenerator")


class BaseWorkloadGenerator(ABC):
    """Base class for all workload generators"""

    def __init__(self,
                 program_name: str,
                 program_source_dir: str,
                 program_build_dir: str,
                 program_compile_dir: str,
                 program_compile_args: str,
                 program_clobber_args: str,
                 output_dir: str = './traces',
                 regression_type: str = 'const_delay',
                 max_attempts: int = 2,
                 iterations: int = 5,
                 num_data_points: int = 500,
                 start_range_counter: int = 0,
                 compress: bool = True):

        self.program_name = program_name
        self.program_source_dir = program_source_dir
        self.program_build_dir = program_build_dir
        self.program_compile_dir = program_compile_dir
        self.program_compile_args = program_compile_args
        self.program_clobber_args = program_clobber_args
        self.output_dir = output_dir
        self.regression_type = regression_type
        self.max_attempts = max_attempts
        self.iterations = iterations
        self.num_data_points = num_data_points
        self.start_range_counter = start_range_counter
        self.compress = compress

        # Set up logger
        self.logger = logger
        self.logger.info(f"Initializing {self.__class__.__name__} for {program_name}")

        # Database manager
        self.db_manager = DBManager()

        # Initialize helpers
        self.regression_manager = RegressionManager(program_name=self.program_name,
                                                    source_dir=self.program_source_dir,
                                                    compile_dir=self.program_compile_dir,
                                                    compile_args=self.program_compile_args,
                                                    clobber_args=self.program_clobber_args)
        self.metrics = MetricsCollector()

        # Set process affinity to first 8 cores
        ResourceManager.set_cpu_affinity()

        # Setup uftrace service
        self.uftrace_service = UftraceService(program_name=self.program_name,
                                              output_dir=self.output_dir,
                                              cwd=self.program_build_dir)

    @abstractmethod
    def prepare_inputs(self, mode: str, from_db: bool = False, db_query: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Prepare inputs based on the execution mode"""
        pass

    @abstractmethod
    def prepare_commands(self, input_data: Any, build_type: str,
                         custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare vanilla and full commands for the given input and candidate functions"""
        pass

    def execute(self,
                mode: str = 'optimized',
                is_regression: bool = False,
                target_regression_function: Optional[str] = None,
                is_baseline: bool = False,
                from_db: bool = False,
                db_query: Optional[Dict[str, Any]] = None,
                rebuild: bool = False,
                without_tracing: bool = False) -> Dict[str, Any]:
        """Execute the workload in the specified mode"""
        self.logger.info(f"Executing {self.program_name} in {mode} mode")
        self.logger.info(f"Regression: {is_regression}, Baseline: {is_baseline}")

        self.metrics.start_execution()

        # Define range clusters and indexes (to be overridden in subclasses if needed)
        range_clusters = ['low', 'mid', 'high'] if is_regression else ['itself']
        range_indexes = ['0', '1', '2', '3', '4'] if is_regression else ['0']

        # Prepare inputs
        inputs = self.prepare_inputs(mode, from_db, db_query)
        self.logger.info(f"Prepared {len(inputs)} inputs")

        range_counter = self.start_range_counter
        for range_c in range_clusters:
            for range_i in range_indexes:
                range_spec = f'{range_c.split("-")[0]}-{range_i}'

                # Skip specific combinations if needed
                if self._should_skip_range(range_spec):
                    self.logger.info(f"Skipping range {range_spec}")
                    continue

                self.logger.info(f"Processing range {range_spec}")

                # Inject regression if needed
                if rebuild:
                    if is_regression:
                        if not target_regression_function:
                            raise ValueError("Target regression function must be specified for regression mode")

                        self.regression_manager.inject_regression(target_function=target_regression_function,
                                                                  regression_type=self.regression_type,
                                                                  language="c",
                                                                  skip_build=False)
                    else:
                        self.regression_manager.inject_regression(target_function="",
                                                                  regression_type="none",
                                                                  language="c",
                                                                  skip_build=False,
                                                                  reset=True)

                if is_baseline or mode != 'regression':
                    range_spec = 'itself'

                # Determine input slice based on mode and program
                if mode == 'analysis':
                    start_index, end_index = 0, self.num_data_points
                else:  # 'regression'
                    inputs_per_range = self.num_data_points
                    start_index = inputs_per_range * range_counter
                    end_index = start_index + inputs_per_range

                # Ensure we have enough inputs
                if start_index >= len(inputs):
                    self.logger.warning(f"Not enough inputs for range {range_spec}, skipping.")
                    continue

                # Adjust end_index if we don't have enough inputs
                end_index = min(end_index, len(inputs))

                failures = {}
                for i, input_data in enumerate(inputs[start_index:end_index]):
                    build_type = "regression" if is_regression else "analysis"
                    if is_baseline:
                        build_type += "_baseline"

                    build = {
                        'type': build_type,
                        'range': range_spec if is_regression else 'itself'
                    }

                    self.logger.info(f"Processing input {start_index + i + 1}/{end_index} for range {range_spec}")

                    is_successful = False
                    current_attempt = 0

                    execution_id = f"{self.program_name.replace('.', '_')}_{build_type}_{range_spec}_input{start_index + i}"

                    while not is_successful and current_attempt < self.max_attempts:
                        try:
                            # Prepare commands and parameters
                            program_command, parameters = self.prepare_commands(input_data, build_type)

                            # Run the program with the commands
                            execution_times = []
                            for iteration in range(self.iterations):
                                self.logger.info(f"\t # Iteration {iteration + 1}/{self.iterations}")

                                self._pre_iteration(input_data)
                                trace_result = self.uftrace_service.trace(program_command=program_command,
                                                                          execution_id=execution_id,
                                                                          execution_mode=mode,
                                                                          parameters=parameters,
                                                                          iteration=iteration,
                                                                          build=build,
                                                                          without_tracing=without_tracing)
                                self._post_iteration(input_data)

                                execution_times.append(trace_result['execution_time'])

                            self.logger.info(f"Average execution time over {self.iterations} iterations: {sum(execution_times) / len(execution_times):.3f}s")
                            self.metrics.record_execution_time(build_type, range_spec, sum(execution_times) / len(execution_times))
                            self.metrics.record_success(build_type, range_spec)

                            self.logger.info(f"Successfully executed {build_type} for range {range_spec}")
                            is_successful = True
                        except Exception as e:
                            self.logger.error(f"Execution failed: {e}")
                            current_attempt += 1

                        if not is_successful:
                            self.logger.warning(f"Failed to run {build_type} for input after {self.max_attempts} attempts")
                            self.metrics.record_failure(build_type, range_spec)

                            if build_type not in failures:
                                failures[build_type] = []
                            failures[build_type].append(input_data)

                        # Perform any cleanup needed after processing an input
                        self._cleanup_after_input(input_data)

                    # Compress the trace data if needed
                    if self.compress:
                        trace_output_dir = os.path.join(self.output_dir, self.program_name, mode, execution_id)
                        self._compress_trace_data(trace_output_dir, f"{trace_output_dir}.zip")

                # Save failures for later analysis
                if failures:
                    failure_file = f'failures.{self.program_name}.{range_spec.replace("-","_")}.json'
                    self.logger.warning(f"Saving {sum(len(fails) for fails in failures.values())} failures to {failure_file}")

                    with open(failure_file, 'w') as f:
                        json.dump(failures, f, indent=4)

                range_counter += 1

        self.metrics.end_execution()
        self.logger.info("Execution completed")

        # Log metrics summary
        summary = self.metrics.get_summary()
        self.logger.info(f"Execution summary: {json.dumps(summary, indent=2)}")

        return summary

    def _should_skip_range(self, range_spec: str) -> bool:
        """Check if a specific range should be skipped"""
        # Override in subclasses for program-specific logic
        return False

    def _cleanup_after_input(self, input_data: Any) -> None:
        """Perform cleanup after processing an input"""
        # Override in subclasses for program-specific cleanup
        pass

    def _pre_iteration(self, input_data: Any) -> None:
        """Perform setup before each iteration"""
        # Override in subclasses for program-specific setup
        pass

    def _post_iteration(self, input_data: Any) -> None:
        """Perform actions after each iteration"""
        # Override in subclasses for program-specific actions
        pass

    def _compress_trace_data(self, input_dir: str, zip_name: str) -> None:
        """Compress the input directory into a zip file and remove the original directory"""
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")

        if not zip_name.endswith(".zip"):
            zip_name += ".zip"

        original_dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, _, filenames in os.walk(input_dir)
                                for filename in filenames)
        
        if os.path.exists(zip_name):
            os.remove(zip_name)

        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, input_dir)
                    zipf.write(abs_path, rel_path)

        compressed_size = os.path.getsize(zip_name)

        def human_readable_size(size_bytes: int) -> str:
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_name[i]}"

        self.logger.info(f"Successfully compressed trace data. Original size: {human_readable_size(original_dir_size)}, Compressed size: {human_readable_size(compressed_size)}")

        shutil.rmtree(input_dir)
