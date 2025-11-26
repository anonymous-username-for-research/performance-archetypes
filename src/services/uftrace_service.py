import subprocess
import tempfile
import time
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ijson

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UftraceService")


class UftraceService:

    def __init__(self, program_name: str, output_dir: str, cwd: Optional[str] = None):
        self.program_name = program_name
        self.output_dir = output_dir
        self.cwd = cwd or os.getcwd()

        self.trace_script_path = str(Path(__file__).with_name('lttng.sh'))
        if not os.path.exists(self.trace_script_path):
            raise FileNotFoundError(f"Trace script not found: {self.trace_script_path}")

    def _process_trace_json(self, input_file) -> None:
        if not os.path.exists(input_file):
            raise Exception(f"Trace JSON file not found: {input_file}")

        def convert_decimal(obj):
            if isinstance(obj, dict):
                return {k: convert_decimal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimal(i) for i in obj]
            elif type(obj).__name__ == 'Decimal':
                return float(obj)
            else:
                return obj

        temp_fd, temp_file = tempfile.mkstemp(suffix='.json')
        try:
            with open(temp_fd, 'w') as temp_out:
                temp_out.write('[')
                first_event = True

                with open(input_file, 'rb') as f:
                    for event in ijson.items(f, 'traceEvents.item'):
                        if event.get('ph') not in ['B', 'E']:
                            continue

                        # Add tid if missing
                        if 'tid' not in event and 'pid' in event:
                            event['tid'] = event['pid']

                        # Remove "args" field to reduce size
                        if 'args' in event:
                            del event['args']

                        if not first_event:
                            temp_out.write(',')

                        json.dump(convert_decimal(event), temp_out)
                        first_event = False

                temp_out.write(']')

            # Replace original file
            os.replace(temp_file, input_file)

        except Exception as e:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            raise Exception(f"Failed to process JSON: {e}")

    def _run_vanilla_execution(self, command, timeout=300) -> tuple[float, subprocess.CompletedProcess]:
        try:
            start_time = time.time()
            process = subprocess.run(command, capture_output=True, cwd=self.cwd, timeout=timeout)
            end_time = time.time()

            if process.returncode != 0:
                stderr = process.stderr.decode('utf-8', errors='replace')
                if stderr.strip():
                    logger.error(f"Vanilla stderr: {stderr}")

                exit()

            return end_time - start_time, process

        except subprocess.TimeoutExpired:
            raise Exception(f"Vanilla execution timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Vanilla execution failed: {e}")

    def _run_instrumented_execution(self, program_command: List[str], execution_id: str, execution_mode: str,
                                    iteration: int, timeout: int = 400) -> tuple[float, str, subprocess.CompletedProcess[bytes]]:
        try:
            output_dir = os.path.join(os.path.abspath(self.output_dir), self.program_name, execution_mode, execution_id, f"iter_{iteration}")
            os.makedirs(output_dir, exist_ok=True)

            command = ['bash', self.trace_script_path, '--session-name', execution_id,
                       '--output-dir', output_dir] + program_command

            start_time = time.time()
            process = subprocess.run(command, capture_output=True, cwd=self.cwd, timeout=timeout)
            execution_time = time.time() - start_time

            if process.returncode != 0:
                stderr_output = process.stderr.decode('utf-8', errors='replace')
                logger.error(f"Instrumented stderr: {stderr_output}")

            absolute_path = os.path.abspath(os.path.join(output_dir, "uftrace.data"))

            return execution_time, absolute_path, process

        except subprocess.TimeoutExpired:
            raise Exception(f"Instrumented execution timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Instrumented execution failed: {e}")

    def trace(self, program_command: List[str], execution_id: str, execution_mode: str,
              parameters: Dict[str, Any], iteration: int, build: Dict[str, str],
              without_tracing: bool = False, timeout: int = 400) -> Dict[str, Any]:
        try:
            if not program_command:
                raise Exception("Invalid program_command: must be a non-empty list")

            if not (build and all(k in build for k in ('type', 'range'))):
                raise Exception("Invalid build configuration: must contain 'type' and 'range'")

            if without_tracing:
                elapsed_time, process = self._run_vanilla_execution(program_command, timeout=timeout)

                return {
                    'build': {'type': build['type'], 'range': build['range']},
                    'times': {'vanilla': elapsed_time},
                    'execution_time': elapsed_time,
                    'parameters': parameters,
                    'success': process.returncode == 0
                }

            exec_time, trace_output_path, exec_process = self._run_instrumented_execution(
                program_command=program_command,
                execution_id=execution_id,
                execution_mode=execution_mode,
                iteration=iteration,
                timeout=timeout)

            return ({
                'build': {'type': build['type'], 'range': build['range']},
                'execution_time': exec_time,
                'parameters': parameters,
                'trace_output_path': trace_output_path,
                'success': exec_process.returncode == 0
            })

        except Exception as e:
            raise Exception(f"Unexpected error during tracing: {e}")

    def dump_trace_data(self, uftrace_data_path: str,
                        output_json_path: Optional[str] = None) -> str:
        if not os.path.exists(uftrace_data_path):
            raise Exception(f"uftrace data path does not exist: {uftrace_data_path}")

        if not output_json_path:
            output_json_path = os.path.join(uftrace_data_path, "uftrace.json")

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        try:
            with open(output_json_path, 'w') as f:
                dump_process = subprocess.run(['uftrace', 'dump', '--chrome', '--demangle=no'],
                                              stdout=f, cwd=uftrace_data_path, timeout=450,
                                              stderr=subprocess.PIPE)
            if dump_process.returncode != 0:
                stderr = dump_process.stderr.decode('utf-8', errors='replace')
                raise Exception(f"uftrace dump failed: {stderr}")

        except subprocess.TimeoutExpired:
            raise Exception("uftrace dump timed out")
        except FileNotFoundError:
            raise Exception("uftrace command not found")
        except Exception as e:
            raise Exception(f"Failed to create trace dump: {e}")

        if not os.path.exists(output_json_path):
            raise Exception(f"Trace output file was not created: {output_json_path}")

        logger.info(f"Trace data dumped to {output_json_path}")

        return output_json_path
