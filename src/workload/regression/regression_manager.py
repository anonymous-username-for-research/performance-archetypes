import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RegressionManager")


class RegressionManager:
    """
    Injects regression patterns based on defined types (CPU, Memory, I/O).
    """

    def __init__(self, program_name: str, source_dir: str, compile_dir: str,
                 compile_args: str, clobber_args: Optional[str] = None):
        """
        Initialize RegressionManager

        Args:
            program_name: Name of the program (for logging).
            source_dir: Path to the program's source code directory.
            compile_dir: Path to the program's build directory.
            compile_args: The build command (e.g., "make -j8").
            clobber_args: The clean command (e.g., "make clean").
        """
        self.program_name = program_name
        self.source_dir = Path(source_dir)
        self.compile_dir = Path(compile_dir)
        self.compile_args = compile_args
        self.clobber_args = clobber_args

        self.valid_regression_types = [
            "cpu_bottleneck",
            "memory_bloat",
            "io_contention"
        ]

        logger.info(f"Initialized RegressionManager for {self.program_name}")

    def _get_element_texts(self, element: Optional[ET.Element]) -> str:
        """Utility function to extract text from XML elements"""
        if element is None:
            return ""
        return re.sub(" +", " ", "".join(element.itertext()).strip().replace("\n", "").replace("\t", ""))

    def _get_regression_code(self,
                             regression_type: str,
                             language: str,
                             custom_params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Generates the raw C/C++ code string and required libraries for a given regression.
        """
        code = ""
        libraries = []

        if regression_type == "cpu_bottleneck":
            iterations = custom_params.get("iterations", 50000)
            code = f"""// ----Begin CPU Bottleneck Injection----
volatile int dummy_cpu_reg = 0;
for (int i = 0; i < {iterations}; i++) {{
    dummy_cpu_reg += (i * i) % 123;
}}
// ----End CPU Bottleneck Injection----
            """

        elif regression_type == "memory_bloat":
            size_mb = custom_params.get("size_mb", 100)
            size_bytes = size_mb * 1024 * 1024

            if language == "c":
                libraries = ["stdlib.h", "string.h"]
                code = f"""// ----Begin Memory Bloat Injection----
size_t mem_size_reg = {size_bytes};
char* mem_bloat_reg = (char*) malloc(mem_size_reg);
if (mem_bloat_reg != NULL) {{
    memset(mem_bloat_reg, 0xAA, mem_size_reg);
}}
// ----End Memory Bloat Injection----
                """
            else:  # cpp
                libraries = ["vector", "numeric"]
                code = f"""// ----Begin Memory Bloat Injection----
try {{
    std::vector<char> mem_bloat_reg({size_bytes}, 0xAA);
}} catch (const std::bad_alloc& e) {{
    // Handle allocation failure if necessary
}}
// ----End Memory Bloat Injection----
                """

        elif regression_type == "io_contention":
            iterations = custom_params.get("iterations", 100)
            filepath = custom_params.get("filepath", "/tmp/regression_io.log")

            filepath = filepath.replace("\\", "\\\\")

            if language == "c":
                libraries = ["stdio.h"]
                code = f"""// ----Begin I/O Contention Injection----
for (int i_reg = 0; i_reg < {iterations}; i_reg++) {{
    FILE* f_reg = fopen("{filepath}", "a");
    if (f_reg) {{
        fprintf(f_reg, "Regression I/O line %d\\n", i_reg);
        fclose(f_reg);
    }}
}}
// ----End I/O Contention Injection----
                """
            else:  # cpp
                libraries = ["fstream"]
                code = f"""// ----Begin I/O Contention Injection----
for (int i_reg = 0; i_reg < {iterations}; i_reg++) {{
    std::ofstream f_reg("{filepath}", std::ios_base::app);
    if (f_reg.is_open()) {{
        f_reg << "Regression I/O line " << i_reg << "\\n";
    }}
}}
// ----End I/O Contention Injection----
                """

        return code, libraries

    def _get_code_block_xml(self,
                            regression_type: str,
                            language: str,
                            custom_params: Dict[str, Any]) -> Tuple[List[ET.Element], List[str]]:
        """
        Generates regression code and converts it into a list of srcml XML elements.
        """
        code, libraries = self._get_regression_code(regression_type, language, custom_params)

        if not code:
            logger.error(f"Could not generate code for {regression_type}")
            return [], []

        suffix = ".cpp" if language == "cpp" else ".c"

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(f"void dummy_wrapper_func() {{\n{code}\n}}")
            tmp_path = tmp.name

        xml = ""
        try:
            process = subprocess.run(["srcml", tmp_path], capture_output=True, check=True)
            xml = re.sub('xmlns="[^"]+"', "", process.stdout.decode("utf-8"), count=1)
        except FileNotFoundError:
            logger.error("`srcml` command not found. Please ensure it is installed and in your PATH.")
            return [], []
        except subprocess.CalledProcessError as e:
            logger.error(f"srcml failed on snippet: {e.stderr.decode()}")
            return [], []
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not xml:
            return [], []

        root = ET.fromstring(xml)
        block_content = root.find(".//function/block/block_content")

        if block_content is None:
            logger.error("Could not find block_content in srcml snippet output")
            return [], []

        return list(block_content), libraries

    def _restore_backups(self):
        """Restore source files from backup copies"""
        logger.info(f"Restoring backups in {self.source_dir}...")
        for backup_ext in ["*.backup.cpp", "*.backup.cxx", "*.backup.cc", "*.backup.c"]:
            for p in self.source_dir.rglob(backup_ext):
                original_path = str(p).replace(".backup." + p.suffix[1:], "")
                if os.path.exists(original_path):
                    os.remove(original_path)
                os.rename(p, original_path)
                logger.info(f"Restored {original_path}")

    def _build_program(self, skip_build: bool = False):
        """Build the program after injection"""
        if skip_build:
            logger.info("Skipping build as requested")
            return

        logger.info(f"Building program: {self.program_name}")

        try:
            clobber_cmd = self.clobber_args.split() if self.clobber_args else []
            compile_cmd = self.compile_args.split() if self.compile_args else []

            if not compile_cmd:
                raise ValueError("Compile command is empty")

            # 1. Clean (Clobber)
            if clobber_cmd:
                logger.info(f"Running clobber: {' '.join(clobber_cmd)}")
                proc = subprocess.run(clobber_cmd, cwd=self.compile_dir, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning(f"Clobber command failed: {proc.stderr}")

            # 2. Build (Compile)
            logger.info(f"Running compile: {' '.join(compile_cmd)}")
            proc = subprocess.run(compile_cmd, cwd=self.compile_dir, capture_output=True, text=True, check=True)
            if proc.returncode != 0:
                logger.error(f"Compile command failed: {proc.stderr}")
                raise RuntimeError(f"Compile command failed!\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            logger.info(f"Successfully built {self.program_name}")

        except FileNotFoundError:
            logger.error(f"Build command not found. Check compile/clobber args.")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building {self.program_name}: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Generic error building {self.program_name}: {e}")
            raise

    def inject_regression(self,
                          target_function: str,
                          regression_type: str,
                          language: str,
                          custom_params: Dict[str, Any] = {},
                          reset: bool = False,
                          skip_build: bool = False) -> bool:
        """
        Injects a specified regression into the target function or resets the code.
        """
        try:
            if reset:
                self._restore_backups()
                self._build_program(skip_build)
                logger.info("Successfully reset source code and rebuilt.")
                return True

            if not self._validate_inputs(regression_type, language):
                return False

            success = self._inject_code(
                regression_type,
                target_function,
                language,
                custom_params
            )

            if success:
                self._build_program(skip_build)
                logger.info(f"Successfully injected {regression_type} into {target_function} and rebuilt.")
                return True
            else:
                logger.error(f"Failed to find target function '{target_function}' to inject regression.")

                self._restore_backups()
                self._build_program(skip_build)
                logger.info("Restored code to pre-injection state.")
                return False

        except Exception as e:
            logger.error(f"Error in inject_regression: {e}")
            self._restore_backups()
            logger.info("Restored code due to exception.")
            return False

    def _validate_inputs(self, regression_type: str, language: str) -> bool:
        """Validate input parameters"""
        if regression_type not in self.valid_regression_types:
            logger.error(f"Invalid regression type. Choose from: {', '.join(self.valid_regression_types)}")
            return False

        if language not in ["c", "cpp"]:
            logger.error(f"Invalid language. Choose from: 'c', 'cpp'")
            return False

        if not self.source_dir.is_dir():
            logger.error(f"Program source directory does not exist: {self.source_dir}")
            return False

        if not self.compile_dir.is_dir():
            logger.error(f"Program compile directory does not exist: {self.compile_dir}")
            return False

        return True

    def _inject_code(self, regression_type: str, target_function: str,
                     language: str, custom_params: Dict[str, Any]) -> bool:
        """
        Finds the target function in the source files and injects the code.
        """
        library_base_xml = re.sub('xmlns="[^"]+"', '',
                                  '<unit xmlns="http://www.srcML.org/srcML/src" xmlns:cpp="http://www.srcML.org/srcML/cpp">'
                                  '<cpp:include>#<cpp:directive>include</cpp:directive> <cpp:file>&lt;PUT_HERE&gt;</cpp:file></cpp:include>\n'
                                  '</unit>\n', count=1)

        try:
            code_elements, libraries = self._get_code_block_xml(regression_type, language, custom_params)
        except Exception as e:
            logger.error(f"Failed to generate regression XML: {e}")
            return False

        if not code_elements:
            logger.error("No code elements generated for injection.")
            return False

        affected = False
        file_patterns = ["*.c"] if language == "c" else ["*.cpp", "*.cxx", "*.cc"]

        for ext in file_patterns:
            if affected:
                break

            for p in self.source_dir.rglob(ext):
                if p.name.endswith((".backup.c", ".backup.cpp", ".backup.cxx", ".backup.cc")):
                    continue
                
                try:
                    process = subprocess.run(["srcml", str(p)], capture_output=True, check=True)
                    xml = process.stdout.decode("utf-8")
                    xml = re.sub('xmlns="[^"]+"', "", xml, count=1)

                    original_file_content = Path(p).read_text()

                    root = ET.fromstring(xml)
                    added_libraries = []

                    # 1. Add required libraries
                    for library in libraries:
                        if f"<{library}>" not in self._get_element_texts(root) and library not in added_libraries:
                            elm = ET.fromstring(library_base_xml.replace("PUT_HERE", library))
                            include_elem = elm.find("{http://www.srcML.org/srcML/cpp}include")
                            if include_elem is not None:
                                root.insert(0, include_elem)
                                added_libraries.append(library)

                    # 2. Find and modify target function
                    for function in root.findall(".//function"):
                        function_name = self._get_element_texts(function.find("name"))

                        if function_name != target_function:
                            continue

                        logger.info(f"Found target function '{function_name}' in {p.name}")

                        blk = function.find("block")
                        if blk is None or blk.find("block_content") is None:
                            logger.warning(f"Function {function_name} has no block_content, skipping.")
                            continue

                        block_content = blk.find("block_content")
                        if block_content is None:
                            logger.warning(f"Function {function_name} has empty block, skipping.")
                            continue

                        # 3. Inject code
                        for item in reversed(code_elements):
                            block_content.insert(0, item)

                        affected = True
                        logger.info(f"Injected {regression_type} code into {function_name}.")
                        break

                    if affected:
                        # 4. Write modified XML and convert back to source
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as output_xml:
                            ET.ElementTree(root).write(output_xml, encoding="unicode")
                            output_xml_path = output_xml.name

                        try:
                            backup_path = str(p) + ".backup." + p.suffix[1:]
                            if not os.path.exists(backup_path):
                                Path(backup_path).write_text(original_file_content)
                                os.rename(p, backup_path)

                            subprocess.run(["srcml", output_xml_path, "-o", str(p)], check=True)
                        finally:
                            if os.path.exists(output_xml_path):
                                os.remove(output_xml_path)

                        break

                except subprocess.CalledProcessError as e:
                    logger.error(f"srcml failed on {p}: {e.stderr.decode()}")
                    continue
                except ET.ParseError as e:
                    logger.error(f"Failed to parse XML for {p}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing file {p}: {e}")
                    continue

        return affected

    def get_available_regression_types(self) -> List[str]:
        """Get list of available regression types"""
        return self.valid_regression_types
