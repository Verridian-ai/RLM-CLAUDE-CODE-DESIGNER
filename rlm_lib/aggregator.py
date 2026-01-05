"""
RLM-C Aggregator Module

Handles aggregation of results from sub-agents and cleanup of temporary files.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass
from datetime import datetime
from glob import glob

from .config import RLMConfig, load_config
from .delegator import DelegationResult, TaskStatus


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AggregatedResult:
    """Aggregated result from multiple sub-agent tasks."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    blocked_tasks: int
    total_tokens: int
    total_duration: float
    results: List[DelegationResult]
    summary: Optional[str] = None
    aggregation_timestamp: str = ""

    def __post_init__(self):
        if not self.aggregation_timestamp:
            self.aggregation_timestamp = datetime.now().isoformat()

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.successful_tasks / self.total_tasks) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "total_tokens": self.total_tokens,
            "total_duration": self.total_duration,
            "success_rate": self.success_rate,
            "summary": self.summary,
            "aggregation_timestamp": self.aggregation_timestamp,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Result Loading
# =============================================================================

def _load_result_file(path: Path) -> Optional[DelegationResult]:
    """Load a single result file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DelegationResult.from_dict(data)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        return None


def iterate_results(
    pattern: str = "result_*.json",
    config: Optional[RLMConfig] = None,
) -> Iterator[DelegationResult]:
    """
    Iterate over result files matching a pattern.

    Args:
        pattern: Glob pattern for result files
        config: Optional RLM configuration

    Yields:
        DelegationResult objects
    """
    config = config or load_config()
    results_path = config.get_results_path()

    for result_file in results_path.glob(pattern):
        result = _load_result_file(result_file)
        if result:
            yield result


def load_results(
    pattern: str = "result_*.json",
    config: Optional[RLMConfig] = None,
) -> List[DelegationResult]:
    """
    Load all result files matching a pattern.

    Args:
        pattern: Glob pattern for result files
        config: Optional RLM configuration

    Returns:
        List of DelegationResult objects
    """
    return list(iterate_results(pattern, config))


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_results(
    pattern: str = "result_*.json",
    config: Optional[RLMConfig] = None,
) -> AggregatedResult:
    """
    Aggregate results from multiple sub-agent tasks.

    Args:
        pattern: Glob pattern for result files
        config: Optional RLM configuration

    Returns:
        AggregatedResult with combined metrics
    """
    config = config or load_config()
    results = load_results(pattern, config)

    successful = [r for r in results if r.status == TaskStatus.COMPLETED]
    failed = [r for r in results if r.status == TaskStatus.FAILED]
    blocked = [r for r in results if r.status == TaskStatus.BLOCKED]

    total_tokens = sum(r.tokens_used for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    return AggregatedResult(
        total_tasks=len(results),
        successful_tasks=len(successful),
        failed_tasks=len(failed),
        blocked_tasks=len(blocked),
        total_tokens=total_tokens,
        total_duration=total_duration,
        results=results,
    )


def aggregate_outputs(
    pattern: str = "result_*.json",
    separator: str = "\n\n---\n\n",
    include_metadata: bool = True,
    config: Optional[RLMConfig] = None,
) -> str:
    """
    Aggregate text outputs from sub-agent results into a single string.

    Args:
        pattern: Glob pattern for result files
        separator: String to separate outputs
        include_metadata: Include task metadata in output
        config: Optional RLM configuration

    Returns:
        Combined output string
    """
    config = config or load_config()
    results = load_results(pattern, config)

    outputs = []
    for result in results:
        if result.status == TaskStatus.COMPLETED and result.output:
            if include_metadata:
                header = f"## Task {result.task_id}"
                if result.context_file:
                    header += f" ({Path(result.context_file).name})"
                outputs.append(f"{header}\n\n{result.output}")
            else:
                outputs.append(result.output)

    return separator.join(outputs)


def summarize_results(
    results_file: Optional[str] = None,
    config: Optional[RLMConfig] = None,
) -> str:
    """
    Create a human-readable summary of aggregated results.

    Args:
        results_file: Optional path to a specific results file
        config: Optional RLM configuration

    Returns:
        Summary string
    """
    config = config or load_config()

    if results_file:
        result = _load_result_file(Path(results_file))
        if not result:
            return f"Could not load results from: {results_file}"
        results = [result]
        agg = AggregatedResult(
            total_tasks=1,
            successful_tasks=1 if result.status == TaskStatus.COMPLETED else 0,
            failed_tasks=1 if result.status == TaskStatus.FAILED else 0,
            blocked_tasks=1 if result.status == TaskStatus.BLOCKED else 0,
            total_tokens=result.tokens_used,
            total_duration=result.duration_seconds,
            results=results,
        )
    else:
        agg = aggregate_results(config=config)

    lines = [
        "# RLM-C Results Summary",
        "",
        f"**Total Tasks:** {agg.total_tasks}",
        f"**Successful:** {agg.successful_tasks} ({agg.success_rate:.1f}%)",
        f"**Failed:** {agg.failed_tasks}",
        f"**Blocked:** {agg.blocked_tasks}",
        "",
        f"**Total Tokens:** {agg.total_tokens:,}",
        f"**Total Duration:** {agg.total_duration:.2f}s",
        "",
    ]

    if agg.failed_tasks > 0:
        lines.append("## Failures")
        for r in agg.results:
            if r.status == TaskStatus.FAILED:
                lines.append(f"- Task {r.task_id}: {r.error or 'Unknown error'}")
        lines.append("")

    if agg.blocked_tasks > 0:
        lines.append("## Blocked Tasks")
        for r in agg.results:
            if r.status == TaskStatus.BLOCKED:
                lines.append(f"- Task {r.task_id}: {r.error or 'Unknown reason'}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Export Functions
# =============================================================================

def export_results_json(
    output_path: str | Path,
    pattern: str = "result_*.json",
    config: Optional[RLMConfig] = None,
) -> Path:
    """
    Export aggregated results to a JSON file.

    Args:
        output_path: Path for the output JSON file
        pattern: Glob pattern for result files
        config: Optional RLM configuration

    Returns:
        Path to the created file
    """
    config = config or load_config()
    agg = aggregate_results(pattern, config)

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agg.to_dict(), f, indent=2)

    return output_path


def export_outputs_markdown(
    output_path: str | Path,
    pattern: str = "result_*.json",
    config: Optional[RLMConfig] = None,
) -> Path:
    """
    Export combined outputs to a Markdown file.

    Args:
        output_path: Path for the output Markdown file
        pattern: Glob pattern for result files
        config: Optional RLM configuration

    Returns:
        Path to the created file
    """
    config = config or load_config()

    summary = summarize_results(config=config)
    outputs = aggregate_outputs(pattern, config=config)

    content = f"{summary}\n\n# Task Outputs\n\n{outputs}"

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


# =============================================================================
# Cleanup Functions
# =============================================================================

def cleanup_cache(config: Optional[RLMConfig] = None) -> int:
    """
    Remove all temporary chunk files from the cache directory.

    Args:
        config: Optional RLM configuration

    Returns:
        Number of files removed
    """
    config = config or load_config()
    cache_path = config.get_cache_path()

    if not cache_path.exists():
        return 0

    count = 0
    for chunk_file in cache_path.glob("chunk_*.txt"):
        try:
            chunk_file.unlink()
            count += 1
        except (IOError, OSError):
            pass

    return count


def cleanup_results(
    older_than_hours: Optional[float] = None,
    config: Optional[RLMConfig] = None,
) -> int:
    """
    Remove result files from the results directory.

    Args:
        older_than_hours: Only remove files older than this many hours
        config: Optional RLM configuration

    Returns:
        Number of files removed
    """
    config = config or load_config()
    results_path = config.get_results_path()

    if not results_path.exists():
        return 0

    count = 0
    now = datetime.now()

    for result_file in results_path.glob("result_*.json"):
        try:
            if older_than_hours:
                mtime = datetime.fromtimestamp(result_file.stat().st_mtime)
                age_hours = (now - mtime).total_seconds() / 3600
                if age_hours < older_than_hours:
                    continue

            result_file.unlink()
            count += 1
        except (IOError, OSError):
            pass

    return count


def cleanup_all(config: Optional[RLMConfig] = None) -> Dict[str, int]:
    """
    Remove all temporary files (cache and results).

    Args:
        config: Optional RLM configuration

    Returns:
        Dictionary with counts of removed files by type
    """
    config = config or load_config()

    return {
        "cache_files": cleanup_cache(config),
        "result_files": cleanup_results(config=config),
    }


def get_storage_stats(config: Optional[RLMConfig] = None) -> Dict[str, Any]:
    """
    Get storage statistics for cache and results directories.

    Args:
        config: Optional RLM configuration

    Returns:
        Dictionary with storage statistics
    """
    config = config or load_config()

    def dir_stats(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"exists": False, "files": 0, "size_bytes": 0}

        files = list(path.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "exists": True,
            "files": len(files),
            "size_bytes": total_size,
            "size_human": f"{total_size / 1024:.1f}KB",
        }

    return {
        "cache": dir_stats(config.get_cache_path()),
        "results": dir_stats(config.get_results_path()),
    }
