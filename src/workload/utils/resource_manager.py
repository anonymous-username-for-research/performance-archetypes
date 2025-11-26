from typing import List, Optional
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ResourceManager")


class ResourceManager:
    """Manages system resources for program execution"""

    @staticmethod
    def set_cpu_affinity(cpus: Optional[List[int]] = None) -> None:
        """Set CPU affinity for the current process"""
        if cpus is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                logger.warning("Could not determine CPU count, skipping affinity setting.")
                return

            cpus = list(range(cpu_count))

        try:
            os.sched_setaffinity(0, cpus)
            logger.info(f"Set CPU affinity to cores: {cpus}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
