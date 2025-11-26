import os
from pathlib import Path
import random
import json
import argparse
import shutil
import zipfile

from tmll.tmll_client import TMLLClient

from src.core.critical_path_generator import CriticalPathGenerator
from src.services.uftrace_service import UftraceService
from src.workload.workload_factory import WorkloadFactory


def serialize_dict_to_json(data: dict):
    import pandas as pd

    def convert_timestamps(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_timestamps(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamps(item) for item in obj]
        else:
            return obj

    converted_data = convert_timestamps(data)
    return converted_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=str, required=True, help="Name of the program to trace")
    parser.add_argument("--source-dir", type=str, required=True, help="Path to the program source code")
    parser.add_argument("--build-dir", type=str, required=True, help="Path to the program build directory")
    parser.add_argument("--compile-dir", type=str, default="", help="Path to the program compile directory")
    parser.add_argument("--compile-args", type=str, default="", help="Program compile arguments")
    parser.add_argument("--clobber-args", type=str, default="", help="Clobber compile arguments")
    parser.add_argument("--output", type=str, required=True, help="Directory to store the output traces")
    parser.add_argument("--mode", choices=["analysis", "regression"], default="analysis", help="Execution mode: analysis or regression")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run and trace the program")
    parser.add_argument("--num-data-points", type=int, default=10, help="Number of data points to collect")
    parser.add_argument("--is-regression", action="store_true", help="Enable regression mode")
    parser.add_argument("--regression-type", choices=["time", "memory", "io", "cpu"], default="time", help="Type of regression analysis")
    parser.add_argument("--is-baseline", action="store_true", help="Run as baseline (no regression)")
    parser.add_argument("--from-db", action="store_true", help="Fetch inputs from database")
    parser.add_argument("--db-query", type=str, help="JSON query for database fetch")
    parser.add_argument("--start-range", type=int, default=0, help="Starting range counter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum execution attempts per input")
    parser.add_argument("--compress", action="store_true", help="Compress the output traces")
    parser.add_argument("--without-tracing", action="store_true", help="Run the program without tracing")

    args = parser.parse_args()

    # Set global random seed
    random.seed(args.seed)

    # Parse DB query if provided
    db_query = None
    if args.db_query:
        try:
            db_query = json.loads(args.db_query)
        except json.JSONDecodeError:
            return 1

    # Convert all the paths to absolute paths
    args.source_dir = os.path.abspath(args.source_dir)
    args.build_dir = os.path.abspath(args.build_dir)
    if args.compile_dir:
        args.compile_dir = os.path.abspath(args.compile_dir)
    args.output = os.path.abspath(args.output)

    # Create the workload generator
    workload_generator = WorkloadFactory.create_workload_generator(
        program_name=args.program,
        program_source_dir=args.source_dir,
        program_build_dir=args.build_dir,
        program_compile_dir=args.compile_dir,
        program_compile_args=args.compile_args,
        program_clobber_args=args.clobber_args,
        output_dir=args.output,
        regression_type=args.regression_type,
        max_attempts=args.max_attempts,
        iterations=args.iterations,
        num_data_points=args.num_data_points,
        start_range_counter=args.start_range,
        compress=args.compress
    )

    # Execute the workload
    workload_generator.execute(
        mode=args.mode,
        is_regression=args.is_regression,
        is_baseline=args.is_baseline,
        from_db=args.from_db,
        db_query=db_query,
        without_tracing=args.without_tracing
    )

    # Now, we should iterate over the generated traces and uncompress one by one for further analysis
    mode = args.mode
    program_name = args.program
    output_dir = args.output

    # The files structure is like output_dir/program/mode/PROGRAMNAME_MODE_itself_INPUTX.zip
    traces_dir = os.path.join(output_dir, program_name, mode)
    if not os.path.exists(traces_dir):
        print(f"Traces directory {traces_dir} does not exist!")
        return 1

    trace_files = [f for f in os.listdir(traces_dir) if f.endswith(".zip")]
    if not trace_files:
        print(f"No trace files found in {traces_dir}!")
        return 1

    # Sort by name to have a deterministic order
    trace_files.sort()

    for trace_file in trace_files:
        trace_path = os.path.join(traces_dir, trace_file)
        print(f"Processing trace file: {trace_path}")

        # Get only the name of the file without extension
        file_name = Path(trace_file).stem

        # Quickly check if the "critical_paths.json" already exists for this input
        critical_path_check = os.path.join(os.getcwd(), "critical-paths", program_name, mode, file_name, "iter-2", "critical_paths.json")
        if os.path.exists(critical_path_check) and os.path.getsize(critical_path_check) > 0:
            print(f"Critical path analysis already exists at {critical_path_check}, skipping...")
            continue

        # First remove any existing extracted directory
        extraction_dir = os.path.join(traces_dir, trace_file.replace(".zip", ""))
        if os.path.exists(extraction_dir):
            shutil.rmtree(extraction_dir)

        # Unzip the trace file
        with zipfile.ZipFile(trace_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)

        uftrace_data_dir = os.path.join(traces_dir, trace_file.replace(".zip", ""))
        if not os.path.exists(uftrace_data_dir):
            print(f"Extracted trace file {uftrace_data_dir} does not exist!")
            continue

        # Check how many iterations are there
        # It's expected to have directories like iter_0, iter_1, ..., iter_N
        iter_dirs = [d for d in os.listdir(uftrace_data_dir) if os.path.isdir(os.path.join(uftrace_data_dir, d)) and d.startswith("iter_")]
        if not iter_dirs:
            print(f"No iteration directories found in {uftrace_data_dir}!")

        iter_dirs.sort()  # Sort to have a deterministic order
        for iter_dir in iter_dirs:
            iter_path = os.path.join(uftrace_data_dir, iter_dir)
            iteration_num = iter_dir.split("_")[-1]
            print(f"  Processing iteration {iteration_num} in {iter_path}")

            critical_path_output_path = os.path.join(os.getcwd(), "critical-paths", program_name, mode, file_name, f"iter-{iteration_num}", "critical_paths.json")
            kernel_data_output_path = os.path.join(os.getcwd(), "critical-paths", program_name, mode, file_name, f"iter-{iteration_num}", "kernel_data.json")

            # Check if the output already exists and it's non-empty
            if os.path.exists(critical_path_output_path) and os.path.getsize(critical_path_output_path) > 0 \
               and os.path.exists(kernel_data_output_path) and os.path.getsize(kernel_data_output_path) > 0:
                print(f"  Critical path and kernel data analysis already exists at {critical_path_output_path}, skipping...")
                continue

            uftrace_path = os.path.join(iter_path, "uftrace.data")
            if not os.path.exists(uftrace_path):
                print(f"UFTrace data file {uftrace_path} does not exist!")
                continue

            # Check the size of the uftrace directory. If it's bigger than 2.5GB, skip it as we cannot process it
            def get_dir_size(path):
                total = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total += os.path.getsize(filepath)
                return total

            uftrace_size = get_dir_size(uftrace_path)
            if uftrace_size > 2 * 1024 * 1024 * 1024:
                continue

            try:
                uftrace_service = UftraceService(program_name, os.path.dirname(uftrace_path), os.path.basename(uftrace_path))
                uftrace_json_path = os.path.join(os.path.dirname(uftrace_path), f"{iteration_num}_uftrace.json")
                uftrace_service.dump_trace_data(uftrace_path, uftrace_json_path)
            except Exception as e:
                print(f"Error processing UFTrace data in {uftrace_path}: {e}")
                continue

            lttng_kernel_path = os.path.join(iter_path, "kernel")
            lttng_ust_path = os.path.join(iter_path, "ust", "uid", "1000", "64-bit")

            nametag = trace_file.replace(".zip", "").split("/")[-1] + f"_iter-{iteration_num}"

            traces = [
                {
                    "name": f"trace-kernel-lttng-{nametag}",
                    "path": lttng_kernel_path
                },
                {
                    "name": f"trace-ust-lttng-{nametag}",
                    "path": lttng_ust_path
                },
                {
                    "name": f"trace-uftrace-{nametag}",
                    "path": uftrace_json_path
                }
            ]

            tmll_client = TMLLClient(delete_all=True)
            experiment_name = "exp_" + nametag
            try:
                cpg = CriticalPathGenerator(tmll_client=tmll_client, resample_freq="10us", hotspots_top_n=100)
                critical_paths, kernel_data, function_stats, _ = cpg.get_critical_path(traces=traces,
                                                                                       experiment_name=experiment_name,
                                                                                       top_k_critical_paths=10)

                # Save the critical path analysis results
                os.makedirs(os.path.dirname(critical_path_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(kernel_data_output_path), exist_ok=True)
                json.dump(serialize_dict_to_json(critical_paths), open(critical_path_output_path, "w"), indent=4)
                json.dump(serialize_dict_to_json(kernel_data), open(kernel_data_output_path, "w"), indent=4)
                function_stats_output_path = os.path.join(os.path.dirname(critical_path_output_path), "function_stats.csv")
                function_stats.to_csv(function_stats_output_path, index=False)

                print(f"Critical path analysis saved to {critical_path_output_path}")
                print(f"Kernel data analysis saved to {kernel_data_output_path}")
                print(f"Function stats saved to {function_stats_output_path}")
            except Exception as e:
                print(f"Error processing iteration {iteration_num} in {iter_path}: {e}")
                continue

        # Remove the extracted directory to save space
        if os.path.exists(uftrace_data_dir):
            shutil.rmtree(uftrace_data_dir)
            print(f"Removed extracted directory {uftrace_data_dir} to save space")

    return 0


if __name__ == "__main__":
    exit(main())
