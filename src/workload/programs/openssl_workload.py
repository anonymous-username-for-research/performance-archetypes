import os
import random
import string
from typing import List, Dict, Any, Optional, Tuple

from ..base_workload_generator import BaseWorkloadGenerator


class OpenSSLWorkloadGenerator(BaseWorkloadGenerator):
    """Workload generator for OpenSSL"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Additional directories for OpenSSL
        self.data_dir = os.path.join(self.program_build_dir, "data")
        self.cert_dir = os.path.join(self.program_build_dir, "certs")

        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cert_dir, exist_ok=True)

    def prepare_inputs(self, mode: str, from_db: bool = False, db_query: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Prepare OpenSSL inputs"""
        if from_db and db_query:
            params = self.db_manager.get_previous_parameters(f"{self.program_name}-{mode}", db_query, server=False)

            workloads = []
            for param in params:
                workload = {}   
                for key in ['workload_type', 'workload_id', 'key_bits', 'key_type', 'file_size_kb',
                            'algorithm', 'digest_algorithm', 'encrypt_key', 'days', 'use_salt', 'algorithms',
                            'encoding', 'padding_mode', 'num_iterations', 'prime_bits', 'dh_bits', 'curve_name']:
                    if key in param:
                        workload[key] = param[key]
                workloads.append(workload)

            return workloads
        else:
            workload_types = [
                'certificate',         # Generate key, CSR, and certificate
                'cert_verify',         # Certificate verification operations
                'cert_convert',        # Certificate format conversions
                'encryption',          # Symmetric encrypt/decrypt
                'rsa_encrypt',         # RSA asymmetric encryption
                'signature',           # Sign and verify files
                'digest',              # Generate file digests
                'hmac',                # HMAC operations
                'base64',              # Base64 encoding/decoding
                'key_convert',         # Key format conversions
                'pubkey_ops',          # Public key operations
                'rand',                # Random number generation
                'prime',               # Prime generation and testing
                'pkcs12',              # PKCS#12 operations
                'pkcs7',               # PKCS#7 operations
                'kdf',                 # Key derivation functions
            ]

            compatible_key_types = {
                'certificate': ['rsa', 'ec', 'dsa'],
                'signature': ['rsa', 'ec', 'dsa'],
                'rsa_encrypt': ['rsa']
            }

            compatible_key_bits = {
                'rsa': [512, 1024, 2048, 3072],
                'dsa': [1024, 2048],
                'ec': [256, 384]
            }

            compatible_digests = {
                'rsa': ['sha1', 'sha256', 'sha384', 'sha512'],
                'dsa': ['sha1', 'sha256'],
                'ec': ['sha256', 'sha384', 'sha512']
            }

            compatible_ciphers = [
                'aes-128-cbc', 'aes-192-cbc', 'aes-256-cbc',
                'aes-128-ecb', 'aes-192-ecb', 'aes-256-ecb',
                'aes-128-ctr', 'aes-192-ctr', 'aes-256-ctr',
                'des-ede3-cbc', 'des-ede3'
            ]

            compatible_hmac_digests = ['sha1', 'sha256', 'sha384', 'sha512', 'md5']

            workloads = []
            for i in range(self.num_data_points):
                workload_type = random.choice(workload_types)

                workload = {
                    'workload_type': workload_type,
                    'workload_id': i
                }

                # Add specific parameters based on workload type
                if workload_type == 'certificate':
                    key_type = random.choice(compatible_key_types['certificate'])
                    key_bits = random.choice(compatible_key_bits[key_type])

                    workload.update({
                        'key_bits': key_bits,
                        'key_type': key_type,
                        'encrypt_key': random.choice([True, False]),
                        'days': random.randint(30, 3650)
                    })

                elif workload_type == 'cert_verify':
                    workload.update({
                        'operation': random.choice(['verify', 'fingerprint', 'text', 'subject', 'issuer', 'dates']),
                        'key_type': random.choice(['rsa', 'ec']),
                        'key_bits': random.choice([2048, 3072])
                    })

                elif workload_type == 'cert_convert':
                    workload.update({
                        'source_format': random.choice(['pem', 'der']),
                        'target_format': random.choice(['pem', 'der']),
                        'key_type': random.choice(['rsa', 'ec']),
                        'key_bits': 2048
                    })

                elif workload_type == 'encryption':
                    algorithm = random.choice(compatible_ciphers)
                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'algorithm': algorithm,
                        'use_salt': random.choice([True, False]),
                        'num_iterations': random.choice([1000, 5000, 10000])
                    })

                elif workload_type == 'rsa_encrypt':
                    workload.update({
                        'file_size_kb': random.randint(1, 50),
                        'key_bits': random.choice([512, 1024, 2048]),
                        'padding_mode': random.choice(['pkcs', 'oaep'])
                    })

                elif workload_type == 'signature':
                    key_type = random.choice(compatible_key_types['signature'])
                    key_bits = random.choice(compatible_key_bits[key_type])
                    digest_algorithm = random.choice(compatible_digests[key_type])

                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'key_bits': key_bits,
                        'key_type': key_type,
                        'encrypt_key': random.choice([True, False]),
                        'digest_algorithm': digest_algorithm,
                        'padding_mode': random.choice(['pkcs1', 'pss']) if key_type == 'rsa' else 'default'
                    })

                elif workload_type == 'digest':
                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'algorithms': random.sample(['md5', 'sha1', 'sha256', 'sha384', 'sha512'],
                                                   random.randint(1, 5))
                    })

                elif workload_type == 'hmac':
                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'digest': random.choice(compatible_hmac_digests)
                    })

                elif workload_type == 'base64':
                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'operation': random.choice(['encode', 'decode'])
                    })

                elif workload_type == 'key_convert':
                    key_type = random.choice(['rsa', 'ec', 'dsa'])
                    workload.update({
                        'key_type': key_type,
                        'key_bits': random.choice(compatible_key_bits[key_type]),
                        'source_format': random.choice(['pem', 'der']),
                        'target_format': random.choice(['pem', 'der']),
                        'encrypt_key': random.choice([True, False])
                    })

                elif workload_type == 'pubkey_ops':
                    key_type = random.choice(['rsa', 'ec', 'dsa'])
                    workload.update({
                        'key_type': key_type,
                        'key_bits': random.choice(compatible_key_bits[key_type]),
                        'operation': random.choice(['extract', 'modulus', 'check', 'text'])
                    })

                elif workload_type == 'rand':
                    workload.update({
                        'num_bytes': random.choice([16, 32, 64, 128, 256, 512, 1024]),
                        'encoding': random.choice(['binary', 'base64', 'hex'])
                    })

                elif workload_type == 'prime':
                    workload.update({
                        'prime_bits': random.choice([64, 128, 256, 512, 1024, 2048]),
                        'operation': random.choice(['generate', 'check']),
                        'safe_prime': random.choice([True, False])
                    })

                elif workload_type == 'dhparam':
                    workload.update({
                        'dh_bits': random.choice([256, 512, 1024]),
                        'generator': random.choice([2, 5])
                    })

                elif workload_type == 'pkcs12':
                    workload.update({
                        'key_type': random.choice(['rsa', 'ec']),
                        'key_bits': random.choice([512, 1024, 2048]),
                        'operation': random.choice(['create', 'parse'])
                    })

                elif workload_type == 'pkcs7':
                    workload.update({
                        'file_size_kb': random.randint(10, 1024),
                        'operation': random.choice(['sign', 'verify', 'encrypt', 'decrypt'])
                    })

                elif workload_type == 'kdf':
                    workload.update({
                        'kdf_type': random.choice(['pbkdf2', 'scrypt']),
                        'output_len': random.choice([16, 32, 64]),
                        'iterations': random.choice([1000, 10000, 100000])
                    })

                if workload not in workloads:
                    workloads.append(workload)
                else:
                    i -= 1

            return workloads

    def prepare_commands(self, input_data: Dict[str, Any], build_type: str,
                         custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare OpenSSL commands based on workload type"""
        workload_type = input_data.get('workload_type', 'encryption')
        workload_id = input_data.get('workload_id', 0)

        self.logger.info(f"Preparing commands for workload ID {workload_id} of type {workload_type}")

        try:
            if workload_type == 'certificate':
                return self._prepare_certificate_commands(input_data, build_type, custom_functions)
            elif workload_type == 'cert_verify':
                return self._prepare_cert_verify_commands(input_data, build_type, custom_functions)
            elif workload_type == 'cert_convert':
                return self._prepare_cert_convert_commands(input_data, build_type, custom_functions)
            elif workload_type == 'encryption':
                return self._prepare_encryption_commands(input_data, build_type, custom_functions)
            elif workload_type == 'rsa_encrypt':
                return self._prepare_rsa_encrypt_commands(input_data, build_type, custom_functions)
            elif workload_type == 'signature':
                return self._prepare_signature_commands(input_data, build_type, custom_functions)
            elif workload_type == 'digest':
                return self._prepare_digest_commands(input_data, build_type, custom_functions)
            elif workload_type == 'hmac':
                return self._prepare_hmac_commands(input_data, build_type, custom_functions)
            elif workload_type == 'base64':
                return self._prepare_base64_commands(input_data, build_type, custom_functions)
            elif workload_type == 'key_convert':
                return self._prepare_key_convert_commands(input_data, build_type, custom_functions)
            elif workload_type == 'pubkey_ops':
                return self._prepare_pubkey_ops_commands(input_data, build_type, custom_functions)
            elif workload_type == 'rand':
                return self._prepare_rand_commands(input_data, build_type, custom_functions)
            elif workload_type == 'prime':
                return self._prepare_prime_commands(input_data, build_type, custom_functions)
            elif workload_type == 'dhparam':
                return self._prepare_dhparam_commands(input_data, build_type, custom_functions)
            elif workload_type == 'pkcs12':
                return self._prepare_pkcs12_commands(input_data, build_type, custom_functions)
            elif workload_type == 'pkcs7':
                return self._prepare_pkcs7_commands(input_data, build_type, custom_functions)
            elif workload_type == 'kdf':
                return self._prepare_kdf_commands(input_data, build_type, custom_functions)
            else:
                raise ValueError(f"Unknown workload type: {workload_type}")
        except Exception as e:
            self.logger.error(f"Error preparing commands for workload {workload_id}: {str(e)}")
            return ["echo", f"Error preparing workload {workload_id}: {str(e)}"], {"error": str(e)}

    def _prepare_certificate_commands(self, input_data: Dict[str, Any], build_type: str,
                                      custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for certificate generation"""
        workload_id = input_data.get('workload_id', 0)
        key_bits = input_data.get('key_bits', 2048)
        key_type = input_data.get('key_type', 'rsa')
        encrypt_key = input_data.get('encrypt_key', False)
        days = input_data.get('days', 365)

        config_file = os.path.join(self.data_dir, f"openssl_{workload_id}.cnf")
        key_file = os.path.join(self.cert_dir, f"{key_type}_{key_bits}_{workload_id}.key")
        csr_file = os.path.join(self.cert_dir, f"req_{workload_id}.csr")
        cert_file = os.path.join(self.cert_dir, f"cert_{workload_id}.crt")
        temp_key_file = os.path.join(self.data_dir, f"temp_ec_{workload_id}.pem")
        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")

        password = None
        if encrypt_key:
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            with open(password_file, 'w') as pf:
                pf.write(password)
            os.chmod(password_file, 0o600)

        self._generate_random_config(config_file, key_type, key_bits)

        vanilla_script = os.path.join(self.data_dir, f"cert_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                if encrypt_key and password:
                    f.write(f"./openssl genrsa -passout file:{password_file} -f4 -out {key_file} {key_bits}\n")
                else:
                    f.write(f"./openssl genrsa -f4 -out {key_file} {key_bits}\n")
            elif key_type == 'dsa':
                param_file = os.path.join(self.data_dir, f"dsa_params_{workload_id}.pem")
                f.write(f"./openssl dsaparam -out {param_file} {key_bits}\n")
                if encrypt_key and password:
                    f.write(f"./openssl gendsa -passout file:{password_file} -out {key_file} {param_file}\n")
                else:
                    f.write(f"./openssl gendsa -out {key_file} {param_file}\n")
            elif key_type == 'ec':
                curve = "prime256v1"
                if key_bits == 521:
                    curve = "secp521r1"
                elif key_bits == 384:
                    curve = "secp384r1"
                
                if encrypt_key and password:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {temp_key_file}\n")
                    f.write(f"./openssl ec -in {temp_key_file} -out {key_file} -passout file:{password_file}\n")
                else:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {key_file}\n")

            if encrypt_key and password:
                f.write(f"./openssl req -new -key {key_file} -passin file:{password_file} -out {csr_file} -config {config_file}\n")
                f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -passin file:{password_file} -out {cert_file} -days {days}\n")
            else:
                f.write(f"./openssl req -new -key {key_file} -out {csr_file} -config {config_file}\n")
                f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -out {cert_file} -days {days}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'certificate',
            'config_file': config_file,
            'key_file': key_file,
            'csr_file': csr_file,
            'cert_file': cert_file,
            'key_bits': key_bits,
            'key_type': key_type,
            'encrypt_key': encrypt_key,
            'password': password,
            'password_file': password_file if encrypt_key else None,
            'days': days,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_cert_verify_commands(self, input_data: Dict[str, Any], build_type: str,
                                     custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for certificate verification and inspection"""
        workload_id = input_data.get('workload_id', 0)
        operation = input_data.get('operation', 'verify')
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)

        key_file = os.path.join(self.cert_dir, f"{key_type}_{key_bits}_{workload_id}.key")
        cert_file = os.path.join(self.cert_dir, f"cert_{workload_id}.crt")
        csr_file = os.path.join(self.cert_dir, f"req_{workload_id}.csr")
        config_file = os.path.join(self.data_dir, f"openssl_{workload_id}.cnf")
        output_file = os.path.join(self.data_dir, f"verify_output_{workload_id}.txt")

        self._generate_random_config(config_file, key_type, key_bits)

        vanilla_script = os.path.join(self.data_dir, f"cert_verify_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            elif key_type == 'ec':
                curve = "secp384r1" if key_bits == 384 else "prime256v1"
                f.write(f"./openssl ecparam -name {curve} -genkey -out {key_file}\n")

            f.write(f"./openssl req -new -key {key_file} -out {csr_file} -config {config_file}\n")
            f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -out {cert_file} -days 365\n")

            if operation == 'verify':
                f.write(f"./openssl verify -CAfile {cert_file} {cert_file}\n")
            elif operation == 'fingerprint':
                f.write(f"./openssl x509 -in {cert_file} -noout -fingerprint -sha256 > {output_file}\n")
            elif operation == 'text':
                f.write(f"./openssl x509 -in {cert_file} -noout -text > {output_file}\n")
            elif operation == 'subject':
                f.write(f"./openssl x509 -in {cert_file} -noout -subject > {output_file}\n")
            elif operation == 'issuer':
                f.write(f"./openssl x509 -in {cert_file} -noout -issuer > {output_file}\n")
            elif operation == 'dates':
                f.write(f"./openssl x509 -in {cert_file} -noout -dates > {output_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'cert_verify',
            'operation': operation,
            'cert_file': cert_file,
            'key_file': key_file,
            'output_file': output_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_cert_convert_commands(self, input_data: Dict[str, Any], build_type: str,
                                      custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for certificate format conversion"""
        workload_id = input_data.get('workload_id', 0)
        source_format = input_data.get('source_format', 'pem')
        target_format = input_data.get('target_format', 'der')
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)

        key_file = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.key")
        cert_pem = os.path.join(self.cert_dir, f"cert_{workload_id}.pem")
        cert_der = os.path.join(self.cert_dir, f"cert_{workload_id}.der")
        csr_file = os.path.join(self.cert_dir, f"req_{workload_id}.csr")
        config_file = os.path.join(self.data_dir, f"openssl_{workload_id}.cnf")

        self._generate_random_config(config_file, key_type, key_bits)

        vanilla_script = os.path.join(self.data_dir, f"cert_convert_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            elif key_type == 'ec':
                f.write(f"./openssl ecparam -name prime256v1 -genkey -out {key_file}\n")

            f.write(f"./openssl req -new -key {key_file} -out {csr_file} -config {config_file}\n")
            f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -out {cert_pem} -days 365\n")

            if source_format == 'pem' and target_format == 'der':
                f.write(f"./openssl x509 -in {cert_pem} -outform DER -out {cert_der}\n")
                f.write(f"./openssl x509 -in {cert_der} -inform DER -outform PEM -out {cert_pem}.verify\n")
            elif source_format == 'der' and target_format == 'pem':
                f.write(f"./openssl x509 -in {cert_pem} -outform DER -out {cert_der}\n")
                f.write(f"./openssl x509 -in {cert_der} -inform DER -outform PEM -out {cert_pem}.verify\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'cert_convert',
            'source_format': source_format,
            'target_format': target_format,
            'cert_pem': cert_pem,
            'cert_der': cert_der,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_encryption_commands(self, input_data: Dict[str, Any], build_type: str,
                                     custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for file encryption/decryption"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        algorithm = input_data.get('algorithm', 'aes-256-cbc')
        use_salt = input_data.get('use_salt', True)
        num_iterations = input_data.get('num_iterations', 10000)

        input_file = os.path.join(self.data_dir, f"random_{file_size_kb}kb_{workload_id}.dat")
        encrypted_file = f"{input_file}.enc"
        decrypted_file = f"{encrypted_file}.dec"
        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        with open(password_file, 'w') as pf:
            pf.write(password)
        os.chmod(password_file, 0o600)

        vanilla_script = os.path.join(self.data_dir, f"encrypt_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            enc_cmd = f"./openssl enc -{algorithm} -in {input_file} -out {encrypted_file}"
            enc_cmd += f" -pass file:{password_file} -pbkdf2 -iter {num_iterations}"
            if use_salt:
                enc_cmd += " -salt"
            f.write(f"{enc_cmd}\n")

            dec_cmd = f"./openssl enc -d -{algorithm} -in {encrypted_file} -out {decrypted_file}"
            dec_cmd += f" -pass file:{password_file} -pbkdf2 -iter {num_iterations}"
            if use_salt:
                dec_cmd += " -salt"
            f.write(f"{dec_cmd}\n")

            f.write(f"diff {input_file} {decrypted_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'encryption',
            'input_file': input_file,
            'encrypted_file': encrypted_file,
            'decrypted_file': decrypted_file,
            'algorithm': algorithm,
            'use_salt': use_salt,
            'num_iterations': num_iterations,
            'password': password,
            'password_file': password_file,
            'file_size_kb': file_size_kb,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_rsa_encrypt_commands(self, input_data: Dict[str, Any], build_type: str,
                                     custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for RSA asymmetric encryption/decryption"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 10)
        key_bits = input_data.get('key_bits', 2048)
        padding_mode = input_data.get('padding_mode', 'pkcs')

        input_file = os.path.join(self.data_dir, f"small_{file_size_kb}kb_{workload_id}.dat")
        encrypted_file = f"{input_file}.enc"
        decrypted_file = f"{encrypted_file}.dec"
        key_file = os.path.join(self.cert_dir, f"rsa_{key_bits}_{workload_id}.key")
        pub_key_file = os.path.join(self.cert_dir, f"rsa_{key_bits}_{workload_id}.pub")

        padding_overhead = 66 if padding_mode == 'oaep' else 11
        max_bytes = (key_bits // 8) - padding_overhead
        max_bytes = max(16, max_bytes)
        with open(input_file, 'wb') as f:
            f.write(os.urandom(max_bytes))

        vanilla_script = os.path.join(self.data_dir, f"rsa_encrypt_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            f.write(f"./openssl rsa -in {key_file} -pubout -out {pub_key_file}\n")
            padding_option = '-pkcs' if padding_mode == 'pkcs' else '-oaep'
            f.write(f"./openssl rsautl -encrypt -inkey {pub_key_file} -pubin {padding_option} -in {input_file} -out {encrypted_file}\n")
            f.write(f"./openssl rsautl -decrypt -inkey {key_file} {padding_option} -in {encrypted_file} -out {decrypted_file}\n")
            f.write(f"diff {input_file} {decrypted_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'rsa_encrypt',
            'input_file': input_file,
            'encrypted_file': encrypted_file,
            'decrypted_file': decrypted_file,
            'key_file': key_file,
            'pub_key_file': pub_key_file,
            'key_bits': key_bits,
            'padding_mode': padding_mode,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_signature_commands(self, input_data: Dict[str, Any], build_type: str,
                                    custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for file signing and verification"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)
        encrypt_key = input_data.get('encrypt_key', False)
        digest_algorithm = input_data.get('digest_algorithm', 'sha256')
        padding_mode = input_data.get('padding_mode', 'pkcs1')

        input_file = os.path.join(self.data_dir, f"random_{file_size_kb}kb_{workload_id}.dat")
        sig_file = f"{input_file}.{digest_algorithm}.sig"
        key_file = os.path.join(self.cert_dir, f"{key_type}_{key_bits}_{workload_id}.key")
        pub_key_file = os.path.join(self.data_dir, f"pubkey_{key_type}_{workload_id}.pem")
        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")
        temp_key_file = os.path.join(self.data_dir, f"temp_{key_type}_{workload_id}.pem")

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        password = None
        if encrypt_key:
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            with open(password_file, 'w') as pf:
                pf.write(password)
            os.chmod(password_file, 0o600)

        vanilla_script = os.path.join(self.data_dir, f"sign_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                if encrypt_key and password:
                    f.write(f"./openssl genrsa -passout file:{password_file} -out {key_file} {key_bits}\n")
                else:
                    f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            elif key_type == 'dsa':
                if key_bits not in [1024, 2048, 3072]:
                    key_bits = 2048
                param_file = os.path.join(self.data_dir, f"params_dsa_{workload_id}.pem")
                f.write(f"./openssl dsaparam -out {param_file} {key_bits}\n")
                if encrypt_key and password:
                    f.write(f"./openssl gendsa -out {key_file} -passout file:{password_file} {param_file}\n")
                else:
                    f.write(f"./openssl gendsa -out {key_file} {param_file}\n")
            elif key_type == 'ec':
                curve = "prime256v1"
                if key_bits == 521:
                    curve = "secp521r1"
                elif key_bits == 384:
                    curve = "secp384r1"
                
                if encrypt_key and password:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {temp_key_file}\n")
                    f.write(f"./openssl ec -in {temp_key_file} -out {key_file} -passout file:{password_file}\n")
                else:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {key_file}\n")

            sign_cmd = f"./openssl dgst -{digest_algorithm} -sign {key_file}"
            if encrypt_key and password:
                sign_cmd += f" -passin file:{password_file}"
            if key_type == 'rsa' and padding_mode == 'pss':
                sign_cmd += " -sigopt rsa_padding_mode:pss"
            sign_cmd += f" -out {sig_file} {input_file}"
            f.write(f"{sign_cmd}\n")

            if key_type == 'rsa':
                f.write(f"./openssl rsa -in {key_file} -pubout -out {pub_key_file}")
            elif key_type == 'dsa':
                f.write(f"./openssl dsa -in {key_file} -pubout -out {pub_key_file}")
            elif key_type == 'ec':
                f.write(f"./openssl ec -in {key_file} -pubout -out {pub_key_file}")
            
            if encrypt_key and password:
                f.write(f" -passin file:{password_file}")
            f.write("\n")

            verify_cmd = f"./openssl dgst -{digest_algorithm} -verify {pub_key_file}"
            if key_type == 'rsa' and padding_mode == 'pss':
                verify_cmd += " -sigopt rsa_padding_mode:pss"
            verify_cmd += f" -signature {sig_file} {input_file}"
            f.write(f"{verify_cmd}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'signature',
            'input_file': input_file,
            'sig_file': sig_file,
            'key_file': key_file,
            'pub_key_file': pub_key_file,
            'key_bits': key_bits,
            'key_type': key_type,
            'encrypt_key': encrypt_key,
            'password': password,
            'password_file': password_file if encrypt_key else None,
            'digest_algorithm': digest_algorithm,
            'padding_mode': padding_mode,
            'file_size_kb': file_size_kb,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_digest_commands(self, input_data: Dict[str, Any], build_type: str,
                                 custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for file digest creation"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        algorithms = input_data.get('algorithms', ['md5', 'sha1', 'sha256'])

        input_file = os.path.join(self.data_dir, f"random_{file_size_kb}kb_{workload_id}.dat")

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        vanilla_script = os.path.join(self.data_dir, f"digest_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for algo in algorithms:
                f.write(f"./openssl dgst -{algo} {input_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'digest',
            'input_file': input_file,
            'algorithms': algorithms,
            'file_size_kb': file_size_kb,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_hmac_commands(self, input_data: Dict[str, Any], build_type: str,
                              custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for HMAC operations"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        digest = input_data.get('digest', 'sha256')

        input_file = os.path.join(self.data_dir, f"random_{file_size_kb}kb_{workload_id}.dat")
        key_file = os.path.join(self.data_dir, f"hmac_key_{workload_id}.bin")
        hmac_file = os.path.join(self.data_dir, f"hmac_{workload_id}.txt")

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        with open(key_file, 'wb') as f:
            f.write(os.urandom(32))

        vanilla_script = os.path.join(self.data_dir, f"hmac_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            f.write(f"./openssl dgst -{digest} -hmac \"$(cat {key_file} | ./openssl base64)\" {input_file} > {hmac_file}\n")
            f.write(f"cat {hmac_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'hmac',
            'input_file': input_file,
            'key_file': key_file,
            'hmac_file': hmac_file,
            'digest': digest,
            'file_size_kb': file_size_kb,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_base64_commands(self, input_data: Dict[str, Any], build_type: str,
                                custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for base64 encoding/decoding"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        operation = input_data.get('operation', 'encode')

        input_file = os.path.join(self.data_dir, f"random_{file_size_kb}kb_{workload_id}.dat")
        encoded_file = f"{input_file}.b64"
        decoded_file = f"{encoded_file}.dec"

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        vanilla_script = os.path.join(self.data_dir, f"base64_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            
            if operation == 'encode':
                f.write(f"./openssl base64 -in {input_file} -out {encoded_file}\n")
                f.write(f"./openssl base64 -d -in {encoded_file} -out {decoded_file}\n")
                f.write(f"diff {input_file} {decoded_file}\n")
            else:
                f.write(f"./openssl base64 -in {input_file} -out {encoded_file}\n")
                f.write(f"./openssl base64 -d -in {encoded_file} -out {decoded_file}\n")
                f.write(f"diff {input_file} {decoded_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'base64',
            'input_file': input_file,
            'encoded_file': encoded_file,
            'decoded_file': decoded_file,
            'operation': operation,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_key_convert_commands(self, input_data: Dict[str, Any], build_type: str,
                                     custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for key format conversions"""
        workload_id = input_data.get('workload_id', 0)
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)
        source_format = input_data.get('source_format', 'pem')
        target_format = input_data.get('target_format', 'der')
        encrypt_key = input_data.get('encrypt_key', False)

        key_pem = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.pem")
        key_der = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.der")
        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")

        password = None
        if encrypt_key:
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            with open(password_file, 'w') as pf:
                pf.write(password)
            os.chmod(password_file, 0o600)

        vanilla_script = os.path.join(self.data_dir, f"key_convert_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                if encrypt_key and password:
                    f.write(f"./openssl genrsa -passout file:{password_file} -out {key_pem} {key_bits}\n")
                else:
                    f.write(f"./openssl genrsa -out {key_pem} {key_bits}\n")
            elif key_type == 'ec':
                curve = "prime256v1"
                if key_bits == 384:
                    curve = "secp384r1"
                elif key_bits == 521:
                    curve = "secp521r1"
                
                temp_key = os.path.join(self.data_dir, f"temp_{workload_id}.pem")
                if encrypt_key and password:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {temp_key}\n")
                    f.write(f"./openssl ec -in {temp_key} -out {key_pem} -passout file:{password_file}\n")
                else:
                    f.write(f"./openssl ecparam -name {curve} -genkey -out {key_pem}\n")
            elif key_type == 'dsa':
                if key_bits not in [1024, 2048, 3072]:
                    key_bits = 2048
                param_file = os.path.join(self.data_dir, f"params_{workload_id}.pem")
                f.write(f"./openssl dsaparam -out {param_file} {key_bits}\n")
                if encrypt_key and password:
                    f.write(f"./openssl gendsa -out {key_pem} -passout file:{password_file} {param_file}\n")
                else:
                    f.write(f"./openssl gendsa -out {key_pem} {param_file}\n")

            convert_cmd = f"./openssl {key_type} -in {key_pem} -outform DER -out {key_der}"
            if encrypt_key and password:
                convert_cmd += f" -passin file:{password_file}"
            f.write(f"{convert_cmd}\n")

            back_convert_cmd = f"./openssl {key_type} -in {key_der} -inform DER -outform PEM -out {key_pem}.verify"
            if encrypt_key and password:
                back_convert_cmd += f" -passin file:{password_file}"
            f.write(f"{back_convert_cmd}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'key_convert',
            'key_type': key_type,
            'key_bits': key_bits,
            'source_format': source_format,
            'target_format': target_format,
            'key_pem': key_pem,
            'key_der': key_der,
            'encrypt_key': encrypt_key,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_pubkey_ops_commands(self, input_data: Dict[str, Any], build_type: str,
                                    custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for public key operations"""
        workload_id = input_data.get('workload_id', 0)
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)
        operation = input_data.get('operation', 'extract')

        key_file = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.key")
        pub_key_file = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.pub")
        output_file = os.path.join(self.data_dir, f"pubkey_output_{workload_id}.txt")

        vanilla_script = os.path.join(self.data_dir, f"pubkey_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            elif key_type == 'ec':
                curve = "prime256v1"
                if key_bits == 384:
                    curve = "secp384r1"
                elif key_bits == 521:
                    curve = "secp521r1"
                f.write(f"./openssl ecparam -name {curve} -genkey -out {key_file}\n")
            elif key_type == 'dsa':
                if key_bits not in [1024, 2048, 3072]:
                    key_bits = 2048
                param_file = os.path.join(self.data_dir, f"params_{workload_id}.pem")
                f.write(f"./openssl dsaparam -out {param_file} {key_bits}\n")
                f.write(f"./openssl gendsa -out {key_file} {param_file}\n")

            if operation == 'extract':
                f.write(f"./openssl {key_type} -in {key_file} -pubout -out {pub_key_file}\n")
            elif operation == 'modulus' and key_type == 'rsa':
                f.write(f"./openssl rsa -in {key_file} -noout -modulus > {output_file}\n")
            elif operation == 'check':
                f.write(f"./openssl {key_type} -in {key_file} -check > {output_file}\n")
            elif operation == 'text':
                f.write(f"./openssl {key_type} -in {key_file} -noout -text > {output_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'pubkey_ops',
            'key_type': key_type,
            'key_bits': key_bits,
            'operation': operation,
            'key_file': key_file,
            'pub_key_file': pub_key_file,
            'output_file': output_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_rand_commands(self, input_data: Dict[str, Any], build_type: str,
                              custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for random number generation"""
        workload_id = input_data.get('workload_id', 0)
        num_bytes = input_data.get('num_bytes', 256)
        encoding = input_data.get('encoding', 'hex')

        output_file = os.path.join(self.data_dir, f"random_{workload_id}.{encoding}")

        vanilla_script = os.path.join(self.data_dir, f"rand_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            
            if encoding == 'hex':
                f.write(f"./openssl rand -hex {num_bytes} > {output_file}\n")
            elif encoding == 'base64':
                f.write(f"./openssl rand -base64 {num_bytes} > {output_file}\n")
            else:
                f.write(f"./openssl rand {num_bytes} > {output_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'rand',
            'num_bytes': num_bytes,
            'encoding': encoding,
            'output_file': output_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_prime_commands(self, input_data: Dict[str, Any], build_type: str,
                               custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for prime number generation and testing"""
        workload_id = input_data.get('workload_id', 0)
        prime_bits = input_data.get('prime_bits', 512)
        operation = input_data.get('operation', 'generate')
        safe_prime = input_data.get('safe_prime', False)

        prime_file = os.path.join(self.data_dir, f"prime_{workload_id}.txt")

        vanilla_script = os.path.join(self.data_dir, f"prime_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            
            if operation == 'generate':
                cmd = f"./openssl prime -generate -bits {prime_bits}"
                if safe_prime:
                    cmd += " -safe"
                cmd += f" > {prime_file}"
                f.write(f"{cmd}\n")
            elif operation == 'check':
                cmd = f"./openssl prime -generate -bits {prime_bits}"
                if safe_prime:
                    cmd += " -safe"
                cmd += f" > {prime_file}"
                f.write(f"{cmd}\n")
                f.write(f"PRIME=$(cat {prime_file})\n")
                f.write(f"./openssl prime $PRIME\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'prime',
            'prime_bits': prime_bits,
            'operation': operation,
            'safe_prime': safe_prime,
            'prime_file': prime_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_dhparam_commands(self, input_data: Dict[str, Any], build_type: str,
                                 custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for Diffie-Hellman parameter generation"""
        workload_id = input_data.get('workload_id', 0)
        dh_bits = input_data.get('dh_bits', 1024)
        generator = input_data.get('generator', 2)

        dh_file = os.path.join(self.data_dir, f"dhparam_{dh_bits}_{workload_id}.pem")
        output_file = os.path.join(self.data_dir, f"dhparam_text_{workload_id}.txt")

        vanilla_script = os.path.join(self.data_dir, f"dhparam_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            f.write(f"./openssl dhparam -out {dh_file} {dh_bits}\n")
            f.write(f"./openssl dhparam -in {dh_file} -text -noout > {output_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'dhparam',
            'dh_bits': dh_bits,
            'generator': generator,
            'dh_file': dh_file,
            'output_file': output_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_pkcs12_commands(self, input_data: Dict[str, Any], build_type: str,
                                custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for PKCS#12 operations"""
        workload_id = input_data.get('workload_id', 0)
        key_type = input_data.get('key_type', 'rsa')
        key_bits = input_data.get('key_bits', 2048)
        operation = input_data.get('operation', 'create')

        key_file = os.path.join(self.cert_dir, f"{key_type}_{workload_id}.key")
        cert_file = os.path.join(self.cert_dir, f"cert_{workload_id}.crt")
        csr_file = os.path.join(self.cert_dir, f"req_{workload_id}.csr")
        p12_file = os.path.join(self.cert_dir, f"bundle_{workload_id}.p12")
        config_file = os.path.join(self.data_dir, f"openssl_{workload_id}.cnf")
        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")

        password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        with open(password_file, 'w') as pf:
            pf.write(password)
        os.chmod(password_file, 0o600)

        self._generate_random_config(config_file, key_type, key_bits)

        vanilla_script = os.path.join(self.data_dir, f"pkcs12_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            if key_type == 'rsa':
                f.write(f"./openssl genrsa -out {key_file} {key_bits}\n")
            elif key_type == 'ec':
                f.write(f"./openssl ecparam -name prime256v1 -genkey -out {key_file}\n")

            f.write(f"./openssl req -new -key {key_file} -out {csr_file} -config {config_file}\n")
            f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -out {cert_file} -days 365\n")

            f.write(f"PASSWORD=$(cat {password_file})\n\n")

            if operation == 'create':
                f.write(f"./openssl pkcs12 -export -in {cert_file} -inkey {key_file} -out {p12_file} -password pass:$PASSWORD\n")
            elif operation == 'parse':
                f.write(f"./openssl pkcs12 -export -in {cert_file} -inkey {key_file} -out {p12_file} -password pass:$PASSWORD\n")
                out_cert = os.path.join(self.data_dir, f"parsed_cert_{workload_id}.pem")
                out_key = os.path.join(self.data_dir, f"parsed_key_{workload_id}.pem")
                f.write(f"./openssl pkcs12 -in {p12_file} -passin pass:$PASSWORD -passout pass:$PASSWORD -out {out_cert} -clcerts -nokeys\n")
                f.write(f"./openssl pkcs12 -in {p12_file} -passin pass:$PASSWORD -passout pass:$PASSWORD -out {out_key} -nocerts\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'pkcs12',
            'key_type': key_type,
            'key_bits': key_bits,
            'operation': operation,
            'key_file': key_file,
            'cert_file': cert_file,
            'p12_file': p12_file,
            'password_file': password_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_pkcs7_commands(self, input_data: Dict[str, Any], build_type: str,
                               custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for PKCS#7 operations"""
        workload_id = input_data.get('workload_id', 0)
        file_size_kb = input_data.get('file_size_kb', 100)
        operation = input_data.get('operation', 'sign')

        input_file = os.path.join(self.data_dir, f"data_{file_size_kb}kb_{workload_id}.txt")
        key_file = os.path.join(self.cert_dir, f"rsa_{workload_id}.key")
        cert_file = os.path.join(self.cert_dir, f"cert_{workload_id}.crt")
        csr_file = os.path.join(self.cert_dir, f"req_{workload_id}.csr")
        signed_file = os.path.join(self.data_dir, f"signed_{workload_id}.p7s")
        config_file = os.path.join(self.data_dir, f"openssl_{workload_id}.cnf")

        if not os.path.exists(input_file):
            with open(input_file, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))

        self._generate_random_config(config_file)

        vanilla_script = os.path.join(self.data_dir, f"pkcs7_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")

            f.write(f"./openssl genrsa -out {key_file} 2048\n")
            f.write(f"./openssl req -new -key {key_file} -out {csr_file} -config {config_file}\n")
            f.write(f"./openssl x509 -req -in {csr_file} -signkey {key_file} -out {cert_file} -days 365\n")

            if operation == 'sign':
                f.write(f"./openssl smime -sign -in {input_file} -signer {cert_file} -inkey {key_file} -out {signed_file}\n")
            elif operation == 'verify':
                f.write(f"./openssl smime -sign -in {input_file} -signer {cert_file} -inkey {key_file} -out {signed_file}\n")
                f.write(f"./openssl smime -verify -in {signed_file} -signer {cert_file} -CAfile {cert_file}\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'pkcs7',
            'operation': operation,
            'input_file': input_file,
            'key_file': key_file,
            'cert_file': cert_file,
            'signed_file': signed_file,
            'file_size_kb': file_size_kb,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _prepare_kdf_commands(self, input_data: Dict[str, Any], build_type: str,
                             custom_functions: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Prepare commands for Key Derivation Functions"""
        workload_id = input_data.get('workload_id', 0)
        kdf_type = input_data.get('kdf_type', 'pbkdf2')
        output_len = input_data.get('output_len', 32)
        iterations = input_data.get('iterations', 10000)

        password_file = os.path.join(self.data_dir, f"password_{workload_id}.txt")
        salt_file = os.path.join(self.data_dir, f"salt_{workload_id}.bin")
        output_file = os.path.join(self.data_dir, f"derived_key_{workload_id}.bin")

        password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        with open(password_file, 'w') as pf:
            pf.write(password)
        os.chmod(password_file, 0o600)

        with open(salt_file, 'wb') as sf:
            sf.write(os.urandom(16))

        vanilla_script = os.path.join(self.data_dir, f"kdf_script_{workload_id}.sh")
        with open(vanilla_script, 'w') as f:
            f.write("#!/bin/bash\nset -e\n\n")
            
            if kdf_type == 'pbkdf2':
                salt_hex = ''.join(format(b, '02x') for b in os.urandom(16))
                f.write(f"./openssl kdf -kdfopt digest:SHA256 -kdfopt pass:$(cat {password_file}) -kdfopt salt:{salt_hex} -kdfopt iter:{iterations} -keylen {output_len} PBKDF2 > {output_file}\n")
            elif kdf_type == 'scrypt':
                salt_hex = ''.join(format(b, '02x') for b in os.urandom(16))
                f.write(f"./openssl kdf -kdfopt pass:$(cat {password_file}) -kdfopt salt:{salt_hex} -kdfopt N:1024 -kdfopt r:8 -kdfopt p:1 -keylen {output_len} SCRYPT > {output_file} 2>/dev/null || echo 'scrypt not supported'\n")

        os.chmod(vanilla_script, 0o755)
        program_command = [vanilla_script]

        for function in custom_functions or []:
            program_command.insert(0, '-P')
            program_command.insert(1, function)

        parameters = {
            'workload_type': 'kdf',
            'kdf_type': kdf_type,
            'output_len': output_len,
            'iterations': iterations,
            'password_file': password_file,
            'salt_file': salt_file,
            'output_file': output_file,
            'script_file': vanilla_script,
            'workload_id': workload_id
        }

        return program_command, parameters

    def _generate_random_config(self, filename: str, key_type: str = 'rsa', key_bits: int = 2048) -> None:
        """Generate a random OpenSSL configuration file with compatible digest"""
        if key_type == 'dsa':
            digest_algos = ['sha256']
        elif key_type == 'ec':
            digest_algos = ['sha256', 'sha384', 'sha512']
        else:
            if key_bits < 1024:
                digest_algos = ['sha1', 'sha256']
            elif key_bits < 2048:
                digest_algos = ['sha256']
            else:
                digest_algos = ['sha256', 'sha384', 'sha512']
        
        digest = random.choice(digest_algos)
        key_size = key_bits
        country = ''.join(random.choices(string.ascii_uppercase, k=2))
        state = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))
        locality = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))
        org = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 15)))
        org_unit = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 15)))
        cn = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10))) + '.example.com'

        with open(filename, 'w') as f:
            f.write(f"""
[ req ]
default_bits = {key_size}
default_md = {digest}
distinguished_name = req_dn
prompt = no
encrypt_key = yes
string_mask = utf8only

[ req_dn ]
C = {country}
ST = {state}
L = {locality}
O = {org}
OU = {org_unit}
CN = {cn}

[ server_cert ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
""")

    def _cleanup_after_input(self, input_data: Any) -> None:
        """Clean up after processing a workload"""
        if not isinstance(input_data, dict):
            return

        script_file = input_data.get('script_file')
        if script_file and os.path.exists(script_file):
            os.unlink(script_file)
            self.logger.info(f"Cleaned up script file: {script_file}")

        workload_type = input_data.get('workload_type')

        cleanup_keys = []
        if workload_type in ['certificate', 'cert_verify', 'cert_convert']:
            cleanup_keys = ['config_file', 'key_file', 'csr_file', 'cert_file', 'output_file', 'cert_pem', 'cert_der']
        elif workload_type in ['encryption', 'rsa_encrypt']:
            cleanup_keys = ['input_file', 'encrypted_file', 'decrypted_file', 'key_file', 'pub_key_file']
        elif workload_type == 'signature':
            cleanup_keys = ['input_file', 'sig_file', 'config_file', 'key_file', 'pub_key_file']
        elif workload_type == 'digest':
            cleanup_keys = ['input_file']
        elif workload_type == 'hmac':
            cleanup_keys = ['input_file', 'key_file', 'hmac_file']
        elif workload_type == 'base64':
            cleanup_keys = ['input_file', 'encoded_file', 'decoded_file']
        elif workload_type in ['key_convert', 'pubkey_ops']:
            cleanup_keys = ['key_pem', 'key_der', 'key_file', 'pub_key_file', 'output_file']
        elif workload_type == 'rand':
            cleanup_keys = ['output_file']
        elif workload_type == 'prime':
            cleanup_keys = ['prime_file']
        elif workload_type == 'dhparam':
            cleanup_keys = ['dh_file', 'output_file']
        elif workload_type in ['pkcs12', 'pkcs7']:
            cleanup_keys = ['key_file', 'cert_file', 'p12_file', 'signed_file', 'input_file']
        elif workload_type == 'kdf':
            cleanup_keys = ['password_file', 'salt_file', 'output_file']

        for key in cleanup_keys:
            file_path = input_data.get(key)
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                self.logger.info(f"Cleaned up {key}: {file_path}")