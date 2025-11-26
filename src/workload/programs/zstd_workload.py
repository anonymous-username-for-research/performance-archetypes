import os
import random
import string
from typing import Any, Dict, List, Optional, Tuple

from ..base_workload_generator import BaseWorkloadGenerator


class ZstdWorkloadGenerator(BaseWorkloadGenerator):
    """Workload generator for Zstd"""

    def prepare_inputs(self, mode: str, from_db: bool = False, db_query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Prepare Zstd inputs"""
        if from_db and db_query:
            return self.db_manager.get_previous_parameters(f"{self.program_name}-{mode}", db_query, server=False)
        else:
            operation_types = [
                'compress',          # Standard compression
                'decompress',        # Decompression
                'test',              # Test compressed file integrity
                'compress_stream',   # Streaming compression
                'decompress_stream', # Streaming decompression
                'multi_file',        # Multiple file operations
            ]

            pattern_types = ['random', 'repetitive', 'structured', 'binary', 'text', 'empty', 'small']

            workloads = []
            for i in range(self.num_data_points):
                operation = random.choice(operation_types)

                # File size distribution
                size_choice = random.random()
                if size_choice < 0.3:
                    size_mb = random.uniform(0.001, 0.01)  # 1KB to 10KB
                elif size_choice < 0.8:
                    size_mb = random.uniform(0.01, 0.2)  # 10KB to 200KB
                else:
                    size_mb = random.uniform(0.2, 1)  # 200KB to 1MB

                pattern = random.choice(pattern_types)

                compression_level = random.randint(1, 3)

                if compression_level > 8 and size_mb > 15:
                    size_mb = random.randint(2, 10)
                    compression_level = random.randint(1, 5)

                workload = {
                    'operation': operation,
                    'size_mb': size_mb,
                    'pattern_type': pattern,
                    'compression_level': compression_level,
                    'workload_id': i,
                    'use_threads': False,
                    'num_threads': 0,
                    'use_memory_limit': False,
                    'memory_limit_mb': None,
                    'use_long_distance': False,
                    'long_distance_log': None,
                    'use_checksum': random.choice([True, False, None]),
                    'include_content_size': random.choice([True, False, None]),
                    'format_type': random.choice(['zstd', None]),
                    'use_rsyncable': False,
                    'adapt_mode': False,
                }

                if workload not in workloads:
                    workloads.append(workload)

            return workloads

    def prepare_commands(self, input_data: Dict[str, Any], build_type: str,
                         custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare Zstd commands"""
        operation = input_data.get('operation', 'compress')
        size_mb = input_data.get('size_mb', 5)
        pattern_type = input_data.get('pattern_type', 'random')
        compression_level = input_data.get('compression_level', 3)
        workload_id = input_data.get('workload_id', 0)

        data_dir = os.path.join(self.program_build_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        input_file = os.path.join(data_dir, f"workload_{workload_id}_{pattern_type}_{size_mb}mb.dat")

        parameters = {
            'operation': operation,
            'pattern_type': pattern_type,
            'size_mb': size_mb,
            'compression_level': compression_level,
            'workload_id': workload_id
        }

        if operation == 'compress':
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"
            program_command = self._build_compress_command(input_data, input_file, compressed_file)
            parameters.update({'input_file': input_file, 'compressed_file': compressed_file})

        elif operation == 'decompress':
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"
            decompressed_file = f"{input_file}.decompressed"

            os.system(f"cd {self.program_build_dir} && ./zstd -q -f -3 {input_file} -o {compressed_file}")

            program_command = ['./zstd', '-d', '-f', compressed_file, '-o', decompressed_file]
            parameters.update({
                'input_file': compressed_file,
                'compressed_file': compressed_file,
                'decompressed_file': decompressed_file,
                'original_file': input_file
            })

        elif operation == 'test':
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"
            os.system(f"cd {self.program_build_dir} && ./zstd -q -f -{compression_level} {input_file} -o {compressed_file}")

            program_command = ['./zstd', '-t', compressed_file]
            parameters.update({'input_file': input_file, 'compressed_file': compressed_file})

        elif operation == 'benchmark':
            self._generate_random_file(size_mb, pattern_type, input_file)
            program_command = ['./zstd', '-b', f'-{compression_level}', '-i1', input_file]
            parameters.update({'input_file': input_file})

        elif operation == 'compress_dict':
            dict_file = os.path.join(data_dir, f"dict_{workload_id}.dict")
            training_files = []

            min_training_size = max(0.5, size_mb * 0.2)
            for j in range(5):
                training_file = os.path.join(data_dir, f"train_{workload_id}_{j}.dat")
                self._generate_random_file(min_training_size, pattern_type, training_file)
                training_files.append(training_file)

            train_cmd = f"cd {self.program_build_dir} && ./zstd --train {' '.join(training_files)} -o {dict_file} --maxdict=112640"
            result = os.system(train_cmd)

            if result != 0 or not os.path.exists(dict_file):
                self._generate_random_file(max(0.1, size_mb), pattern_type, input_file)
                compressed_file = f"{input_file}.zst"
                program_command = self._build_compress_command(input_data, input_file, compressed_file)
                parameters.update({'input_file': input_file, 'compressed_file': compressed_file, 'training_files': training_files})
            else:
                self._generate_random_file(max(0.1, size_mb), pattern_type, input_file)
                compressed_file = f"{input_file}.zst"
                program_command = ['./zstd', f'-{compression_level}', '-D', dict_file, '-f', input_file, '-o', compressed_file]
                parameters.update({
                    'input_file': input_file,
                    'compressed_file': compressed_file,
                    'dict_file': dict_file,
                    'training_files': training_files
                })

        elif operation == 'train_dict':
            dict_file = os.path.join(data_dir, f"dict_{workload_id}.dict")
            training_files = []

            num_samples = random.randint(5, 8)
            sample_size = random.uniform(0.001, 0.01)

            for j in range(num_samples):
                training_file = os.path.join(data_dir, f"train_{workload_id}_{j}.dat")
                self._generate_random_file(sample_size, pattern_type, training_file)

                if os.path.exists(training_file) and os.path.getsize(training_file) > 0:
                    training_files.append(training_file)

            if len(training_files) < 5:
                for j in range(len(training_files), 10):
                    training_file = os.path.join(data_dir, f"train_{workload_id}_extra_{j}.dat")
                    self._generate_random_file(0.05, pattern_type, training_file)
                    if os.path.exists(training_file):
                        training_files.append(training_file)

            dict_size = random.choice([16384, 32768, 65536, 112640])
            program_command = ['./zstd', '--train'] + training_files + ['-o', dict_file, f'--maxdict={dict_size}']

            parameters.update({
                'dict_file': dict_file,
                'training_files': training_files,
                'dict_size': dict_size,
                'num_samples': len(training_files)
            })

        elif operation == 'compress_stream':
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"

            program_command = ['sh', '-c', f'cd {self.program_build_dir} && cat {input_file} | ./zstd -{compression_level} -c > {compressed_file}']
            parameters.update({'input_file': input_file, 'compressed_file': compressed_file, 'streaming': True})

        elif operation == 'decompress_stream':
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"
            decompressed_file = f"{input_file}.decompressed"

            os.system(f"cd {self.program_build_dir} && ./zstd -q -f -3 {input_file} -o {compressed_file}")

            program_command = ['sh', '-c', f'cd {self.program_build_dir} && cat {compressed_file} | ./zstd -d -c > {decompressed_file}']
            parameters.update({
                'input_file': compressed_file,
                'compressed_file': compressed_file,
                'decompressed_file': decompressed_file,
                'streaming': True
            })

        elif operation == 'multi_file':
            input_files = []
            for j in range(2):
                file_name = os.path.join(data_dir, f"multi_{workload_id}_{j}.dat")
                self._generate_random_file(max(0.1, size_mb / 2), pattern_type, file_name)
                input_files.append(file_name)

            program_command = ['./zstd', f'-{compression_level}', '-f'] + input_files
            parameters.update({'input_files': input_files, 'num_files': 2})

        else:
            self._generate_random_file(size_mb, pattern_type, input_file)
            compressed_file = f"{input_file}.zst"
            program_command = ['./zstd', f'-{compression_level}', '-f', input_file, '-o', compressed_file]
            parameters.update({'input_file': input_file, 'compressed_file': compressed_file})

        program_command = self._add_advanced_options(program_command, input_data)

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        input_data.update(parameters)
        return program_command, parameters

    def _build_compress_command(self, input_data: Dict[str, Any], input_file: str, output_file: str) -> List[str]:
        """Build compression command with various options"""
        compression_level = input_data.get('compression_level', 3)

        cmd = ['./zstd']

        if compression_level > 19:
            cmd.extend(['--ultra', f'-{compression_level}'])
        else:
            cmd.append(f'-{compression_level}')

        cmd.extend(['-f', input_file, '-o', output_file])

        return cmd

    def _add_advanced_options(self, command: List[str], input_data: Dict[str, Any]) -> List[str]:
        """Add advanced Zstd options to command"""
        if 'sh' in command or command[0] == 'sh':
            return command

        insert_idx = 1

        if input_data.get('use_threads') and input_data.get('num_threads', 0) > 0:
            command.insert(insert_idx, f"-T{input_data['num_threads']}")
            insert_idx += 1

        if input_data.get('use_memory_limit') and input_data.get('memory_limit_mb'):
            command.insert(insert_idx, f"-M{input_data['memory_limit_mb']}MB")
            insert_idx += 1

        if input_data.get('use_long_distance') and input_data.get('long_distance_log'):
            command.insert(insert_idx, f"--long={input_data['long_distance_log']}")
            insert_idx += 1

        if input_data.get('use_checksum') is False:
            command.insert(insert_idx, '--no-check')
            insert_idx += 1

        if input_data.get('include_content_size') is False:
            command.insert(insert_idx, '--no-content-size')
            insert_idx += 1

        format_type = input_data.get('format_type')
        if format_type and format_type != 'zstd':
            command.insert(insert_idx, f'--format={format_type}')
            insert_idx += 1

        if input_data.get('use_rsyncable'):
            command.insert(insert_idx, '--rsyncable')
            insert_idx += 1

        if input_data.get('adapt_mode'):
            command.insert(insert_idx, '--adapt')
            insert_idx += 1

        return command

    def _generate_random_file(self, size_mb: float, pattern_type: str, filename: str) -> None:
        """Helper method to generate random files with different patterns"""
        if size_mb == 0:
            open(filename, 'w').close()
            return

        size_bytes = int(size_mb * 1024 * 1024)

        if size_bytes < 1024:
            with open(filename, 'wb') as f:
                if pattern_type == 'random':
                    f.write(os.urandom(size_bytes))
                else:
                    f.write(b'a' * size_bytes)
            return

        with open(filename, 'wb') as f:
            chunk_size = min(10 * 1024 * 1024, size_bytes)
            remaining_bytes = size_bytes

            while remaining_bytes > 0:
                current_chunk_size = min(chunk_size, remaining_bytes)

                if pattern_type == 'random':
                    f.write(os.urandom(current_chunk_size))

                elif pattern_type == 'repetitive':
                    block_size = random.randint(100, 1000)
                    random_block = os.urandom(block_size)

                    bytes_written = 0
                    while bytes_written < current_chunk_size:
                        bytes_to_write = min(block_size, current_chunk_size - bytes_written)
                        f.write(random_block[:bytes_to_write])
                        bytes_written += bytes_to_write

                    remaining_bytes -= current_chunk_size
                    continue

                elif pattern_type == 'structured':
                    template_open = b'{"id": %d, "name": "%s", "values": ['
                    template_close = b']}\n'

                    bytes_written = 0
                    record_id = 0

                    while bytes_written < current_chunk_size:
                        name_length = random.randint(5, 15)
                        name = ''.join(random.choices(string.ascii_letters, k=name_length)).encode('utf-8')

                        record = template_open % (record_id, name.decode('utf-8').encode('utf-8'))

                        values = []
                        for _ in range(random.randint(5, 20)):
                            values.append(str(random.randint(0, 1000)).encode('utf-8'))

                        record += b', '.join(values)
                        record += template_close

                        if bytes_written + len(record) <= current_chunk_size:
                            f.write(record)
                            bytes_written += len(record)
                            record_id += 1
                        else:
                            padding = current_chunk_size - bytes_written
                            f.write(b' ' * padding)
                            bytes_written = current_chunk_size

                elif pattern_type == 'binary':
                    header = b'BINARY' + bytes([random.randint(0, 255) for _ in range(10)])
                    f.write(header)
                    bytes_written = len(header)

                    struct_size = random.randint(16, 64)

                    while bytes_written < current_chunk_size:
                        struct = bytearray([0] * struct_size)

                        struct[0:4] = [0xDE, 0xAD, 0xBE, 0xEF]

                        struct[4:8] = [random.randint(0, 255) & 0xF0 for _ in range(4)]

                        struct[8:12] = [(bytes_written // struct_size) & 0xFF,
                                        ((bytes_written // struct_size) >> 8) & 0xFF,
                                        ((bytes_written // struct_size) >> 16) & 0xFF,
                                        ((bytes_written // struct_size) >> 24) & 0xFF]

                        for i in range(12, struct_size):
                            struct[i] = random.randint(0, 255)

                        to_write = min(struct_size, current_chunk_size - bytes_written)
                        f.write(struct[:to_write])
                        bytes_written += to_write

                elif pattern_type == 'small':
                    f.write(b'SMALL_FILE_CONTENT_' * (current_chunk_size // 19))
                    f.write(b'SMALL_FILE_CONTENT_'[:(current_chunk_size % 19)])

                elif pattern_type == 'empty':
                    f.write(b'\x00' * current_chunk_size)

                else:
                    words = []
                    for _ in range(1000):
                        word_len = random.randint(3, 10)
                        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
                        words.append(word)

                    common_words = words[:100]

                    text = bytearray()
                    while len(text) < current_chunk_size:
                        if random.random() < 0.7:
                            word = random.choice(common_words)
                        else:
                            word = random.choice(words)

                        if random.random() < 0.15:
                            if random.random() < 0.7:
                                word += '.'
                            else:
                                word += random.choice(',;:!?')

                            if random.random() < 0.9:
                                word += ' '

                            if random.random() < 0.1:
                                word += '\n\n'
                        else:
                            word += ' '

                        text.extend(word.encode('utf-8'))

                    f.write(text[:current_chunk_size])

                remaining_bytes -= current_chunk_size

    def _cleanup_after_input(self, input_data: Any) -> None:
        """Clean up generated files after processing"""
        parameters = input_data if isinstance(input_data, dict) else {}

        files_to_cleanup = []

        if 'input_file' in parameters:
            files_to_cleanup.append(parameters['input_file'])
        if 'compressed_file' in parameters:
            files_to_cleanup.append(parameters['compressed_file'])
        if 'decompressed_file' in parameters:
            files_to_cleanup.append(parameters['decompressed_file'])
        if 'original_file' in parameters:
            files_to_cleanup.append(parameters['original_file'])
        if 'dict_file' in parameters:
            files_to_cleanup.append(parameters['dict_file'])
        if 'training_files' in parameters:
            files_to_cleanup.extend(parameters['training_files'])
        if 'input_files' in parameters:
            files_to_cleanup.extend(parameters['input_files'])
            for f in parameters['input_files']:
                files_to_cleanup.append(f + '.zst')

        for file_path in files_to_cleanup:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                    self.logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up file {file_path}: {e}")
