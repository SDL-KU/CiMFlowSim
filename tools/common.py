#!/usr/bin/env python3
"""
Common utilities for tools/*.py scripts.

This module provides shared functionality to ensure consistency across all CLI tools.
SINGLE SOURCE OF TRUTH for:
- Python path setup
- Signal handlers for graceful shutdown
- Project directory paths
"""

import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Any

# Setup Python path for all tools
PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_SRC = PROJECT_ROOT / "src" / "python"
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


class TaskRegistry:
    """
    Registry for tracking active Ray tasks.

    Replaces global variable pattern with a proper class that can be
    safely passed to signal handlers and accessed from multiple locations.
    """

    def __init__(self) -> None:
        self._tasks: List[Any] = []

    def add(self, task: Any) -> None:
        """Add a task to the registry."""
        self._tasks.append(task)

    def remove(self, task: Any) -> None:
        """Remove a task from the registry."""
        try:
            self._tasks.remove(task)
        except ValueError:
            pass  # Task not in list

    def clear(self) -> None:
        """Clear all tasks from the registry."""
        self._tasks.clear()

    def cancel_all(self) -> int:
        """
        Cancel all registered Ray tasks.

        Returns:
            Number of tasks that were canceled
        """
        if not self._tasks:
            return 0

        count = 0
        try:
            import ray


            for task in self._tasks:
                try:
                    ray.cancel(task, force=True)
                    count += 1
                except Exception:
                    pass
            self._tasks.clear()
        except Exception:
            pass
        return count

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)


# Global task registry instance (singleton pattern)
_task_registry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    """Get the global task registry instance."""
    return _task_registry


def cleanup_and_exit(ray_tasks: TaskRegistry | None = None):
    """
    Clean up all worker processes and exit.

    Args:
        ray_tasks: TaskRegistry instance or None (uses global registry)
    """
    print("\n\nInterrupt received, cleaning up workers...")

    # Use global registry if no argument provided
    if ray_tasks is None:
        ray_tasks = _task_registry

    # Cancel Ray tasks
    if len(ray_tasks) > 0:
        print(f"Canceling {len(ray_tasks)} Ray tasks...")
        canceled = ray_tasks.cancel_all()
        print(f"{canceled} Ray tasks canceled")

    # Kill only child processes of this process (not other users' processes)
    current_pid = os.getpid()
    subprocess.run(["pkill", "-9", "-P", str(current_pid)], stderr=subprocess.DEVNULL)

    # Shutdown Ray if initialized
    try:
        import ray
        if ray.is_initialized():
            print("Shutting down Ray...")
            ray.shutdown()
            print("Ray shutdown complete")
    except Exception:
        pass

    print("Workers cleaned up")
    sys.exit(0)


def setup_signal_handlers(ray_tasks: TaskRegistry | None = None):
    """
    Setup signal handlers for graceful shutdown.

    Args:
        ray_tasks: TaskRegistry instance or None (uses global registry)
    """
    # Use global registry if no argument provided
    if ray_tasks is None:
        ray_tasks = _task_registry

    def handler(signum, frame):
        cleanup_and_exit(ray_tasks)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def setup_ray_cleanup():
    """Register atexit handler to shutdown Ray on program exit."""
    def cleanup_on_exit():
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    atexit.register(cleanup_on_exit)
