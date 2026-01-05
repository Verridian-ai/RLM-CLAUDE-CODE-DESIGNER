"""
RLM-C Kernel Module

The main orchestrator for the Recursive Language Model system.
Provides a unified interface for processing large contexts.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import (
    RLMConfig,
    load_config,
    MAX_FILE_SIZE_BYTES,
    ENV_RLM_MODE,
    ENV_CTX_FILE,
    ENV_RLM_STATE_FILE,
    is_rlm_mode_active,
    get_state_file,
)
from .chunker import (
    chunk_data,
    get_file_info,
    preview_file,
    detect_content_type,
    ChunkInfo,
    FileInfo,
)
from .delegator import (
    delegate_task,
    delegate_task_batch,
    check_recursion_depth,
    get_token_budget_remaining,
    get_token_budget_status,
    reset_token_budget,
    DelegationResult,
    TaskStatus,
)
from .aggregator import (
    aggregate_results,
    aggregate_outputs,
    summarize_results,
    cleanup_cache,
    cleanup_results,
    cleanup_all,
    get_storage_stats,
    AggregatedResult,
)
from .indexer import (
    search_with_ripgrep,
    build_index,
    search_index,
    find_files,
    SearchResult,
)


# =============================================================================
# Kernel State
# =============================================================================

class KernelStatus(str, Enum):
    """Status of the RLM Kernel."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class KernelState:
    """Persistent state for the RLM Kernel."""
    status: KernelStatus = KernelStatus.UNINITIALIZED
    context_file: Optional[str] = None
    chunks_created: int = 0
    tasks_delegated: int = 0
    tokens_used: int = 0
    last_activity: str = ""
    errors: List[str] = field(default_factory=list)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "context_file": self.context_file,
            "chunks_created": self.chunks_created,
            "tasks_delegated": self.tasks_delegated,
            "tokens_used": self.tokens_used,
            "last_activity": self.last_activity,
            "errors": self.errors,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelState":
        data["status"] = KernelStatus(data.get("status", "uninitialized"))
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "KernelState":
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return cls()


# =============================================================================
# RLM Kernel
# =============================================================================

class RLMKernel:
    """
    The RLM Kernel - main orchestrator for recursive language model operations.

    This class provides a unified interface for:
    - Processing large files that exceed context limits
    - Delegating tasks to sub-agents
    - Managing chunking and aggregation
    - Enforcing safety guardrails
    """

    def __init__(self, config: Optional[RLMConfig] = None):
        """
        Initialize the RLM Kernel.

        Args:
            config: Optional RLM configuration
        """
        self.config = config or load_config()
        self._state_file = get_state_file()
        self._state = KernelState.load(self._state_file)
        self._current_chunks: List[ChunkInfo] = []

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self, context_file: Optional[str] = None) -> "RLMKernel":
        """
        Initialize the kernel for a new session.

        Args:
            context_file: Optional default context file path

        Returns:
            Self for chaining
        """
        import uuid

        self.config.ensure_directories()

        self._state = KernelState(
            status=KernelStatus.READY,
            context_file=context_file,
            session_id=str(uuid.uuid4())[:8],
            last_activity=datetime.now().isoformat(),
        )

        # Set environment variables
        os.environ[ENV_RLM_MODE] = "active"
        if context_file:
            os.environ[ENV_CTX_FILE] = context_file

        self._save_state()
        return self

    def _save_state(self) -> None:
        """Save current state to disk."""
        self._state.last_activity = datetime.now().isoformat()
        self._state.save(self._state_file)

    def _log_error(self, error: str) -> None:
        """Log an error to state."""
        self._state.errors.append(f"{datetime.now().isoformat()}: {error}")
        if len(self._state.errors) > 100:
            self._state.errors = self._state.errors[-50:]
        self._save_state()

    # =========================================================================
    # Status and Validation
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get current kernel status and statistics.

        Returns:
            Dictionary with status information
        """
        can_recurse, depth, max_depth = check_recursion_depth()
        budget = get_token_budget_status()
        storage = get_storage_stats(self.config)

        return {
            "kernel_status": self._state.status.value,
            "session_id": self._state.session_id,
            "context_file": self._state.context_file,
            "rlm_mode_active": is_rlm_mode_active(),
            "recursion": {
                "current_depth": depth,
                "max_depth": max_depth,
                "can_recurse": can_recurse,
            },
            "budget": budget,
            "storage": storage,
            "statistics": {
                "chunks_created": self._state.chunks_created,
                "tasks_delegated": self._state.tasks_delegated,
                "tokens_used": self._state.tokens_used,
            },
            "last_activity": self._state.last_activity,
            "error_count": len(self._state.errors),
        }

    def validate_operation(
        self,
        tool: str,
        args: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if an operation is allowed under RLM constraints.

        Args:
            tool: Name of the tool being used
            args: Arguments to the tool

        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check for direct file read operations
        if tool.lower() in ("read", "cat", "head", "tail"):
            file_path = args.get("file_path") or args.get("path") or args.get("file")
            if file_path:
                try:
                    info = get_file_info(file_path)
                    if info.size_bytes > self.config.max_file_size_bytes:
                        return False, (
                            f"File too large for direct context ({info.size_human}). "
                            f"Use kernel.process_query() or chunk_data() instead."
                        )
                except FileNotFoundError:
                    pass  # Let the tool handle missing files

        # Check recursion depth
        can_recurse, depth, max_depth = check_recursion_depth()
        if tool.lower() in ("delegate", "delegate_task", "task"):
            if not can_recurse:
                return False, f"Recursion limit reached: {depth}/{max_depth}"

        # Check token budget
        if tool.lower() in ("delegate", "delegate_task", "task"):
            remaining = get_token_budget_remaining()
            if remaining < 1000:
                return False, f"Token budget nearly exhausted: {remaining} remaining"

        return True, None

    def requires_rlm(self, file_path: str | Path) -> bool:
        """
        Check if a file requires RLM processing (too large for direct read).

        Args:
            file_path: Path to check

        Returns:
            True if file requires RLM, False if it can be read directly
        """
        try:
            info = get_file_info(file_path)
            return info.size_bytes > self.config.max_file_size_bytes
        except FileNotFoundError:
            return False

    # =========================================================================
    # Core Processing
    # =========================================================================

    def process_query(
        self,
        query: str,
        context_file: Optional[str] = None,
        model: str = "haiku",
        chunk_strategy: str = "auto",
    ) -> AggregatedResult:
        """
        Process a query against a large context file using RLM.

        This is the main entry point for RLM processing. It:
        1. Chunks the context file if needed
        2. Delegates each chunk to a sub-agent
        3. Aggregates the results

        Args:
            query: The question/task to perform
            context_file: Path to the large context file
            model: Model to use for sub-agents
            chunk_strategy: "auto", "code", "document", or "line"

        Returns:
            AggregatedResult with combined sub-agent outputs
        """
        self._state.status = KernelStatus.PROCESSING
        self._save_state()

        context_file = context_file or self._state.context_file
        if not context_file:
            self._log_error("No context file specified")
            raise ValueError("No context file specified. Provide context_file parameter or set via initialize().")

        context_path = Path(context_file)
        if not context_path.exists():
            self._log_error(f"Context file not found: {context_file}")
            raise FileNotFoundError(f"Context file not found: {context_file}")

        try:
            # Get file info
            info = get_file_info(context_path)

            # If file is small enough, process directly
            if info.size_bytes <= self.config.max_file_size_bytes:
                result = delegate_task(
                    instruction=query,
                    context_file=str(context_path),
                    model=model,
                    config=self.config,
                )
                self._state.tasks_delegated += 1
                self._state.tokens_used += result.tokens_used
                self._save_state()

                return AggregatedResult(
                    total_tasks=1,
                    successful_tasks=1 if result.status == TaskStatus.COMPLETED else 0,
                    failed_tasks=1 if result.status == TaskStatus.FAILED else 0,
                    blocked_tasks=1 if result.status == TaskStatus.BLOCKED else 0,
                    total_tokens=result.tokens_used,
                    total_duration=result.duration_seconds,
                    results=[result],
                )

            # Chunk the file
            chunks = chunk_data(context_path, config=self.config)
            self._current_chunks = chunks
            self._state.chunks_created += len(chunks)

            # Create tasks for each chunk
            tasks = []
            for i, chunk in enumerate(chunks):
                chunk_query = (
                    f"Processing chunk {i+1}/{len(chunks)} "
                    f"(lines {chunk.start_line}-{chunk.end_line}).\n\n"
                    f"Task: {query}"
                )
                tasks.append({
                    "instruction": chunk_query,
                    "context_file": str(chunk.chunk_path),
                    "model": model,
                })

            # Delegate to sub-agents
            results = delegate_task_batch(tasks, config=self.config)
            self._state.tasks_delegated += len(results)
            self._state.tokens_used += sum(r.tokens_used for r in results)
            self._save_state()

            # Aggregate results
            aggregated = AggregatedResult(
                total_tasks=len(results),
                successful_tasks=len([r for r in results if r.status == TaskStatus.COMPLETED]),
                failed_tasks=len([r for r in results if r.status == TaskStatus.FAILED]),
                blocked_tasks=len([r for r in results if r.status == TaskStatus.BLOCKED]),
                total_tokens=sum(r.tokens_used for r in results),
                total_duration=sum(r.duration_seconds for r in results),
                results=results,
            )

            self._state.status = KernelStatus.READY
            self._save_state()

            return aggregated

        except Exception as e:
            self._state.status = KernelStatus.ERROR
            self._log_error(str(e))
            raise

    def search(
        self,
        query: str,
        search_path: Optional[str] = None,
        max_results: int = 20,
        context_lines: int = 2,
    ) -> SearchResult:
        """
        Search for text in files using RLM-safe methods.

        Args:
            query: Search query (regex supported)
            search_path: Path to search in (default: current directory)
            max_results: Maximum results to return
            context_lines: Lines of context around matches

        Returns:
            SearchResult with matches
        """
        search_path = search_path or str(Path.cwd())
        return search_with_ripgrep(
            query=query,
            search_path=search_path,
            max_results=max_results,
            context_lines=context_lines,
        )

    def preview(
        self,
        file_path: str,
        lines: int = 50,
        from_end: bool = False,
    ) -> str:
        """
        Safely preview a file's content.

        Args:
            file_path: Path to the file
            lines: Number of lines to preview
            from_end: Show last N lines instead of first N

        Returns:
            Preview string
        """
        return preview_file(file_path, lines, from_end)

    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            FileInfo with metadata
        """
        return get_file_info(file_path)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self, include_results: bool = False) -> Dict[str, int]:
        """
        Clean up temporary files.

        Args:
            include_results: Also remove result files

        Returns:
            Dictionary with counts of removed files
        """
        counts = {"cache_files": cleanup_cache(self.config)}

        if include_results:
            counts["result_files"] = cleanup_results(config=self.config)

        # Clear chunk references
        self._current_chunks = []

        return counts

    def reset(self) -> None:
        """Reset the kernel to initial state."""
        self.cleanup(include_results=True)
        reset_token_budget()
        self._state = KernelState()
        self._save_state()


# =============================================================================
# Module-level convenience instance
# =============================================================================

_kernel: Optional[RLMKernel] = None


def get_kernel() -> RLMKernel:
    """Get or create the global RLM Kernel instance."""
    global _kernel
    if _kernel is None:
        _kernel = RLMKernel()
    return _kernel


def init_kernel(context_file: Optional[str] = None) -> RLMKernel:
    """Initialize and return the global RLM Kernel."""
    global _kernel
    _kernel = RLMKernel()
    _kernel.initialize(context_file)
    return _kernel


# =============================================================================
# Convenience Functions
# =============================================================================

def process_query(
    query: str,
    context_file: Optional[str] = None,
    model: str = "haiku",
) -> AggregatedResult:
    """
    Convenience function to process a query using the global kernel.

    Args:
        query: The question/task
        context_file: Path to context file
        model: Model for sub-agents

    Returns:
        AggregatedResult
    """
    kernel = get_kernel()
    if kernel._state.status == KernelStatus.UNINITIALIZED:
        kernel.initialize(context_file)
    return kernel.process_query(query, context_file, model)


def status() -> Dict[str, Any]:
    """Get status of the global kernel."""
    return get_kernel().get_status()


def validate(tool: str, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate an operation against RLM constraints."""
    return get_kernel().validate_operation(tool, args)
