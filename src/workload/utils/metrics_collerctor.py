import time
from typing import Any, Dict


class MetricsCollector:
    """Collects execution metrics and statistics"""

    def __init__(self):
        self.execution_times = {}
        self.success_counts = {}
        self.failure_counts = {}
        self.start_time = None
        self.end_time = None

    def start_execution(self) -> None:
        """Mark the start of execution"""
        self.start_time = time.time()

    def end_execution(self) -> None:
        """Mark the end of execution"""
        self.end_time = time.time()

    def record_success(self, candidate_type: str, range_spec: str) -> None:
        """Record a successful execution"""
        key = (candidate_type, range_spec)
        self.success_counts[key] = self.success_counts.get(key, 0) + 1

    def record_failure(self, candidate_type: str, range_spec: str) -> None:
        """Record a failed execution"""
        key = (candidate_type, range_spec)
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

    def record_execution_time(self, candidate_type: str, range_spec: str, duration: float) -> None:
        """Record execution time for a specific candidate type and range"""
        key = (candidate_type, range_spec)
        if key not in self.execution_times:
            self.execution_times[key] = []
        self.execution_times[key].append(duration)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of execution metrics"""
        if self.start_time is None or self.end_time is None:
            return {"error": "Execution not complete"}

        total_duration = self.end_time - self.start_time
        total_success = sum(self.success_counts.values())
        total_failure = sum(self.failure_counts.values())

        metrics = {
            "total_duration_seconds": total_duration,
            "total_success": total_success,
            "total_failure": total_failure,
            "success_rate": total_success / (total_success + total_failure) if (total_success + total_failure) > 0 else 0,
        }

        times_by_type = {}
        for (candidate_type, range_spec), times in self.execution_times.items():
            if times:
                times_by_type[f"{candidate_type}_{range_spec}"] = {
                    "min": min(times),
                    "max": max(times),
                    "avg": sum(times) / len(times)
                }

        if times_by_type:
            metrics["execution_times_by_type"] = times_by_type

        return metrics
