# Performance Archetypes: Multi-Execution Critical Path Pattern Analysis for C/C++ Applications

This repository is the replication package for the *Performance Archetypes: Multi-Execution Critical Path Pattern Analysis for C/C++ Applications* study, submitted to ICPE 2026.

## Overview

1. **Workload Execution & Tracing** – Programs (SQLite, OpenSSL, FFmpeg, Zstd) are compiled, exercised under diverse workloads, and traced with LTTng and uftrace.
2. **Critical-Path Extraction** – The raw traces are processed with the TMLL client to extract function-level critical paths, kernel telemetry, and hotspot statistics.
3. **Static Inspection** – srcML-based analysis generates structural metrics (loops, branches, IO calls, etc.) per function.
4. **Research Questions (RQ1–RQ3)** – Dedicated scripts combine static and dynamic views to answer the research questions in the paper and regenerate the artifacts in `results/`.

## Architecture at a Glance

```
fetch_programs.sh ──▶ program sources
        │
        ▼
src.workload.* ──▶ controlled executions ──▶ traces (uftrace + LTTng)
        │                                          │
        ▼                                          ▼
src.services.UftraceService            src.core.CriticalPathGenerator
        │                                          │
        └───────────────► critical-paths/ ◄────────┘
                                 │
                                 ▼
        static analysis (src.analysis.static_code_analyzer)
                                 │
                                 ▼
                         src.research.rq{1,2,3}
                                 │
                                 ▼
                            results/rq{1,2,3}
```

## Repository Layout

| Path | Description |
| --- | --- |
| `fetch_programs.sh` | Script for cloning/updating program sources. |
| `results/` | Research questions' artifacts (reports, CSVs, PDFs). |
| `statistical-analysis/` | Static metrics JSON for each code base, used by RQ scripts. |
| `libraries/` | Third-party wheels (i.e., `TMLL`) to keep provenance self-contained. |
| `src/` | All Python modules (see below). |
| `critical-paths/` | Experiment outputs written by the pipeline. |

### Key Modules under `src/`

- **`src/core`**
  - `pipeline.py` – CLI entry point that drives the entire data-collection loop.
  - `critical_path_generator.py` – Wrapper around TMLL to compute critical paths, kernel stats, and function hotspots per experiment.

- **`src/services`**
  - `uftrace_service.py` – Encapsulates instrumented runs, JSON conversion, and the `lttng.sh` orchestration script.
  - `lttng.sh` – Shell script configuring kernel and ust sessions.

- **`src/workload`**
  - `workload_factory.py` – Resolves program name to a generator.
  - `base_workload_generator.py` – Shared execution harness (input slicing, retries, compression, regression hooks).
  - `programs/*.py` – Program-specific workloads (for input generation, program execution, and instrumentation)
  - `regression/regression_manager.py` – Injects regressions into C/C++ sources via srcML.

- **`src/analysis`** – `static_code_analyzer.py` builds structural metrics consumed downstream.
- **`src/research`** – `rq1.py`, `rq2.py`, `rq3.py` regenerate the analyses described in the paper.

## Pipeline Components in Detail

| Stage | Responsibility | Relevant Files |
| --- | --- | --- |
| Input preparation | Program-specific generation of command lines and inputs | `src/workload/programs/*.py` |
| Instrumented execution | Runs programs with/without tracing, compresses trace output, etc. | `src/workload/base_workload_generator.py`, `src/services/uftrace_service.py`, `src/services/lttng.sh` |
| Regression injection | Injects CPU, memory, or I/O regressions. | `src/workload/regression/regression_manager.py` |
| Trace post-processing | Converts raw traces into Chrome JSON, feeds TMLL, exports critical paths and kernel metrics. | `src/services/uftrace_service.py`, `src/core/critical_path_generator.py` |
| Analysis & reporting | Correlates static/dynamic metrics, produces plots, reports, etc. | `src/research/rq1.py`, `src/research/rq2.py`, `src/research/rq3.py`, `results/` |

## Getting Started

1. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install libraries/tmll-0.0.40-py3-none-any.whl
   ```
2. **Program Sources** – Use `fetch_programs.sh` to fetch and build the programs.

## Running the Instrumentation Pipeline

```bash
python -m src.core.pipeline \
    --program sqlite \
    --source-dir /path/to/sqlite/src \
    --build-dir /path/to/sqlite/build \
    --compile-dir /path/to/sqlite/build \
    --compile-args "make -j8 all" \
    --clobber-args "make clean" \
    --output ./critical-paths \
    --mode analysis \
    --iterations 3 \
    --num-data-points 500
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--program` | One of `sqlite`, `openssl`, `ffmpeg`, `zstd`. |
| `--mode` | `analysis` (baseline) or `regression` (with injected faults). |
| `--is-regression`, `--regression-type` | Enable and configure synthetic regressions. |
| `--compress` | Store each traced input as a `.zip` for better storage management. |

Outputs land in `critical-paths/<program>/<mode>/...` and include:

- `critical_paths.json` – top-k critical paths from TMLL,
- `kernel_data.json` – CPU/memory/disk utilization series,
- `function_stats.csv` – aggregated hotspot metrics.

## Static Analysis

`src/analysis/static_code_analyzer.py` parses C/C++ sources via srcML to compute per-function metrics (LOC, loops, nesting, callers, branch counts, IO calls). Results live in `statistical-analysis/<program>/analysis/statistical_function_analysis.json`. Re-run it whenever you change program sources to keep RQ scripts in sync.

## Reproducing Research Questions

Each RQ script can be executed independently from the repository root. They expect:

- `critical-paths/` – dynamic data produced by the pipeline,
- `statistical-analysis/` – static metrics,
- `results/` – destination for regenerated figures/reports.

### RQ1 – Static-Dynamic Paradox

```
python -m src.research.rq1 sqlite openssl zstd ffmpeg --base-path .
```

### RQ2 – Performance Archetypes

```
python -m src.research.rq2 sqlite openssl zstd ffmpeg --base-path .
```

### RQ3 – Regression Detection

```
python -m src.research.rq3 sqlite openssl zstd ffmpeg --base-path .
```

## Dataset

The complete dataset is available at [doi.org/10.5281/zenodo.17736094](https://doi.org/10.5281/zenodo.17736094). Note that the extracted dataset exceeds 12GB in size.

A sample subset of critical paths is provided in the [critical-paths-samples](./critical-paths-samples/) folder within this repository for convenience.