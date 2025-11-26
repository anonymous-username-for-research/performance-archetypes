import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET


class CodeAnalyzer:
    """
    Analyzes C/C++ source code to extract function-level metrics and characteristics.
    """

    IO_OPERATIONS = {
        'fopen', 'freopen', 'fclose', 'fflush', 'fwide', 'setbuf', 'setvbuf',
        'fread', 'fwrite',
        'getc', 'fgetc', 'fgets', 'putc', 'fputc', 'fputs', 'getchar', 'gets',
        'putchar', 'puts', 'ungetc',
        'fgetwc', 'getwc', 'fgetws', 'fputwc', 'putwc', 'fputws', 'ungetwc',
        'getwchar', 'putwchar',
        'scanf', 'fscanf', 'sscanf', 'printf', 'fprintf', 'sprintf', 'snprintf',
        'vprintf', 'vfprintf', 'vsprintf', 'vsnprintf', 'vscanf', 'vfscanf', 'vsscanf',
        'wscanf', 'fwscanf', 'swscanf', 'wprintf', 'fwprintf', 'swprintf',
        'vfwprintf', 'vswprintf', 'vwprintf', 'vwscanf', 'vfwscanf', 'vswscanf',
        'ftell', 'fseek', 'rewind', 'fgetpos', 'fsetpos',
        'clearerr', 'feof', 'ferror', 'perror',
        'remove', 'rename', 'tmpfile', 'tmpnam', 'tmpnam_r',
        'read', 'write', 'open', 'close', 'lseek', 'ioctl',
        'cin', 'cout', 'cerr', 'clog', 'getline', 'ifstream', 'ofstream'
    }

    FILE_EXTENSIONS = ['*.cpp', '*.cxx', '*.cc', '*.c', '*.h', '*.hpp', '*.hxx']

    def __init__(self, input_directory: str, function_filter: Optional[List[str]] = None):
        """
        Initialize the CodeAnalyzer.

        Args:
            input_directory: Path to the directory containing source code
            function_filter: Optional list of function names to analyze
        """
        self.input_directory = Path(input_directory)
        self.function_filter = set([f.lower() for f in function_filter]) if function_filter else None
        self.result = []
        self.callers = {}
        self.function_count = 0
        self.filtered_count = 0

        if not self.input_directory.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")

        try:
            subprocess.run(['srcml', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("srcml is not installed or not available in PATH")

    @staticmethod
    def get_element_texts(element: Optional[ET.Element] = None) -> str:
        """
        Extract and clean text content from an XML element.

        Args:
            element: XML element to extract text from

        Returns:
            Cleaned text content
        """
        if element is None:
            return ''
        text = ''.join(element.itertext()).strip().replace('\n', '').replace('\t', '')
        return re.sub(' +', ' ', text)

    @staticmethod
    def extract_simple_function_name(full_name: str) -> str:
        """
        Extract the simple function name from a full function signature.

        Args:
            full_name: Full function name possibly including return type and parameters

        Returns:
            Simple function name
        """
        # Remove everything after the first opening parenthesis
        name = full_name.split('(')[0].strip()

        # Handle scope resolution (e.g., std::vector::push_back -> push_back)
        if '::' in name:
            name = name.split('::')[-1]

        # Remove return type and modifiers (take last word)
        parts = name.split()
        if len(parts) > 0:
            name = parts[-1]

        # Remove any pointer/reference symbols
        name = name.lstrip('*&')

        return name.strip()

    def should_analyze_function(self, function_name: str) -> bool:
        """
        Determine if a function should be analyzed based on the filter.

        Args:
            function_name: Full function name

        Returns:
            True if function should be analyzed, False otherwise
        """
        if self.function_filter is None:
            return True

        simple_name = self.extract_simple_function_name(function_name)
        return simple_name.lower() in self.function_filter

    def parse_file_with_srcml(self, file_path: Path) -> Optional[ET.Element]:
        """
        Parse a source file using srcml.

        Args:
            file_path: Path to the source file

        Returns:
            Root element of the parsed XML tree
        """
        try:
            process = subprocess.run(
                ['srcml', str(file_path)],
                capture_output=True,
                check=True
            )
            xml = process.stdout.decode('utf-8')
            xml = re.sub('xmlns="[^"]+"', '', xml, count=1)
            return ET.fromstring(xml)
        except subprocess.CalledProcessError as e:
            return None
        except ET.ParseError as e:
            return None

    def analyze_function(self, function: ET.Element, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a single function and extract its metrics.

        Args:
            function: XML element representing the function
            file_path: Path to the source file containing the function

        Returns:
            Dictionary containing function metrics
        """
        function_name = self.get_element_texts(function.find('name'))

        # Count parameters
        number_of_parameters = len(function.findall('.//parameter'))

        # Find loops
        loops = (function.findall('.//for') +
                 function.findall('.//while') +
                 function.findall('.//do'))
        number_of_loops = len(loops)

        # Count nested loops
        number_of_nested_loops = 0
        for loop in loops:
            nested_loops = (loop.findall('.//for') +
                            loop.findall('.//while') +
                            loop.findall('.//do'))
            number_of_nested_loops += len(nested_loops)

        # Calculate lines of code (semicolons + blocks)
        number_of_semicolons = ''.join(function.itertext()).count(';')

        # Subtract semicolons in loop controls
        for loop in loops:
            control = loop.find('control')
            if control is not None:
                number_of_semicolons -= ''.join(control.itertext()).count(';')

        number_of_blocks = len(function.findall('.//block'))
        line_of_codes = number_of_semicolons + number_of_blocks - 1

        # Check for I/O operations
        has_io = any(
            self.get_element_texts(call.find('name')) in self.IO_OPERATIONS
            for call in function.findall('.//call')
        )

        # Analyze function calls
        callees = []
        is_recursive = False
        simple_function_name = self.extract_simple_function_name(function_name)

        for call in function.findall('.//call'):
            callee_name = self.get_element_texts(call.find('name'))
            callees.append(callee_name)

            # Check for recursion using simple name
            if self.extract_simple_function_name(callee_name) == simple_function_name:
                is_recursive = True

            self.callers[callee_name] = self.callers.get(callee_name, 0) + 1

        # Count statement types
        number_of_expression_statements = len(function.findall('.//expr_stmt'))
        number_of_declaration_statements = len(function.findall('.//decl_stmt'))
        number_of_empty_statements = len(function.findall('.//empty_stmt'))

        # Count branches
        number_of_if = len(function.findall('.//if') + function.findall('.//else'))
        number_of_switch = len(function.findall('.//switch'))
        number_of_preprocessor_if = len(
            function.findall('.//if') +
            function.findall('.//else') +
            function.findall('.//elif')
        )

        return {
            'name': function_name,
            'simple_name': simple_function_name,
            'number_of_parameters': number_of_parameters,
            'line_of_codes': line_of_codes,
            'has_io': has_io,
            'number_of_loops': number_of_loops,
            'number_of_nested_loops': number_of_nested_loops,
            'number_of_calls': callees,
            'is_recursive': is_recursive,
            'number_of_statements': {
                'number_of_expression_statements': number_of_expression_statements,
                'number_of_declaration_statements': number_of_declaration_statements,
                'number_of_empty_statements': number_of_empty_statements
            },
            'number_of_branches': {
                'number_of_if': number_of_if,
                'number_of_switch': number_of_switch,
                'number_of_preprocessor_if': number_of_preprocessor_if
            }
        }

    def analyze_directory(self) -> List[Dict[str, Any]]:
        """
        Analyze all source files in the input directory.

        Returns:
            List of file analysis results
        """
        for ext in self.FILE_EXTENSIONS:
            for file_path in self.input_directory.rglob(ext):
                root = self.parse_file_with_srcml(file_path)

                if root is None:
                    continue

                relative_path = str(file_path.relative_to(self.input_directory))

                file_entry = next(
                    (item for item in self.result if item['file'] == relative_path),
                    None
                )

                functions_in_file = []
                for function in root.findall('.//function'):
                    self.function_count += 1
                    function_data = self.analyze_function(function, file_path)

                    # Check if function should be included based on filter
                    if self.should_analyze_function(function_data['name']):
                        functions_in_file.append(function_data)
                        self.filtered_count += 1

                # Only add file entry if it has functions matching the filter
                if functions_in_file:
                    if file_entry is None:
                        file_entry = {
                            'file': relative_path,
                            'functions': []
                        }
                        self.result.append(file_entry)

                    file_entry['functions'].extend(functions_in_file)

        return self.result

    def finalize_analysis(self) -> None:
        """Finalize the analysis by adding caller counts."""
        for file_entry in self.result:
            for function in file_entry['functions']:
                function['number_of_callers'] = self.callers.get(function['name'], 0)
                function['number_of_calls'] = len(function['number_of_calls'])

    def save_results(self, output_file: str = 'result.json') -> None:
        """Save analysis results to a JSON file."""
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.result, f, indent=4)

    def run(self) -> List[Dict[str, Any]]:
        """Run the complete analysis pipeline."""
        self.analyze_directory()
        self.finalize_analysis()
        return self.result


def main():
    parser = argparse.ArgumentParser(description='Analyze C/C++ source code and extract function-level metrics')

    parser.add_argument(
        'input_directory',
        help='Path to the directory containing C/C++ source code'
    )

    parser.add_argument(
        '-f', '--functions',
        nargs='+',
        metavar='FUNCTION',
        help='List of function names to analyze (simple names only, e.g., main analyze)'
    )

    parser.add_argument(
        '-o', '--output',
        default='result.json',
        metavar='FILE',
        help='Output JSON file path (default: result.json)'
    )

    args = parser.parse_args()

    try:
        analyzer = CodeAnalyzer(args.input_directory, args.functions)
        analyzer.run()
        analyzer.save_results(args.output)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
