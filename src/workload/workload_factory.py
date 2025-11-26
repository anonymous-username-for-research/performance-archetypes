from .base_workload_generator import BaseWorkloadGenerator
from .programs.openssl_workload import OpenSSLWorkloadGenerator
from .programs.zstd_workload import ZstdWorkloadGenerator
from .programs.sqlite_workload import SQLiteWorkloadGenerator
from .programs.ffmpeg_workload import FFmpegWorkloadGenerator


class WorkloadFactory:
    """Factory for creating workload generators based on program name"""

    @staticmethod
    def create_workload_generator(program_name: str, **kwargs) -> BaseWorkloadGenerator:
        """Create a workload generator for the specified program"""
        # Define a mapping of program names to their generator classes
        program_generators = {
            "openssl": OpenSSLWorkloadGenerator,
            "zstd": ZstdWorkloadGenerator,
            "sqlite": SQLiteWorkloadGenerator,
            "ffmpeg": FFmpegWorkloadGenerator,
        }

        if program_name not in program_generators:
            raise ValueError(f"Unsupported program: {program_name}")

        # Create and return the appropriate generator
        generator_class = program_generators[program_name]
        return generator_class(
            program_name=program_name,
            program_source_dir=kwargs.get('program_source_dir', ''),
            program_build_dir=kwargs.get('program_build_dir', ''),
            program_compile_dir=kwargs.get('program_compile_dir', ''),
            program_compile_args=kwargs.get('program_compile_args', ''),
            program_clobber_args=kwargs.get('program_clobber_args', False),
            output_dir=kwargs.get('output_dir', './traces'),
            regression_type=kwargs.get('regression_type', 'const_delay'),
            max_attempts=kwargs.get('max_attempts', 2),
            iterations=kwargs.get('iterations', 5),
            num_data_points=kwargs.get('num_data_points', 500),
            start_range_counter=kwargs.get('start_range_counter', 0),
            compress=kwargs.get('compress', True)
        )
