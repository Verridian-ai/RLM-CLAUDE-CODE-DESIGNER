"""
RLM-C Configuration Module

Defines constants and configuration for the RLM system.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Core Constants
# =============================================================================

# File size threshold - files larger than this MUST use RLM tools
MAX_FILE_SIZE_BYTES: int = 512_000  # 500KB

# Maximum tokens per chunk (approximate - assumes ~4 chars per token)
MAX_CHUNK_SIZE_TOKENS: int = 5_000
MAX_CHUNK_SIZE_CHARS: int = MAX_CHUNK_SIZE_TOKENS * 4  # ~20KB

# Recursion limits to prevent infinite loops / fork bombs
MAX_RECURSION_DEPTH: int = 3

# Token budget per session (for cost control)
MAX_TOKEN_BUDGET: int = 100_000

# Directory names for caching and results
CACHE_DIR: str = ".cache"
RESULTS_DIR: str = "results"

# =============================================================================
# Multi-Agent Architecture Configuration
# =============================================================================

# Claude 4.5 Model Configuration
# All models now use Claude 4.5 with consistent naming

# Main orchestrator model (Opus for complex reasoning and decision-making)
DEFAULT_ORCHESTRATOR_MODEL: str = "claude-opus-4-5-20250514"

# Default model for sub-agents (Sonnet for coding tasks)
DEFAULT_SUBAGENT_MODEL: str = "claude-sonnet-4-5-20250514"

# Escalation model (Opus for when Sonnet struggles)
ESCALATION_MODEL: str = "claude-opus-4-5-20250514"

# Fast model for simple extraction tasks
FAST_MODEL: str = "claude-haiku-4-5-20250514"

# Enable ultra-thinking mode for main orchestrator by default
ENABLE_ULTRA_THINKING: bool = True

# Maximum retries before escalating to a more capable model
MAX_RETRIES_BEFORE_ESCALATION: int = 2

# Indicators that a sub-agent is struggling (triggers escalation)
STRUGGLE_INDICATORS: list = [
    "I'm unable to",
    "I cannot determine",
    "insufficient information",
    "beyond my capabilities",
    "need more context",
    "error",
    "failed",
    "cannot complete",
]

# Chunk overlap (lines) for context continuity
CHUNK_OVERLAP_LINES: int = 10

# File extensions for content type detection
CODE_EXTENSIONS: set = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
    ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".sass", ".less",
}

DOCUMENT_EXTENSIONS: set = {
    ".txt", ".md", ".markdown", ".rst", ".rtf", ".doc", ".docx",
    ".pdf", ".tex", ".org", ".adoc", ".asciidoc",
}

LOG_EXTENSIONS: set = {
    ".log", ".out", ".err", ".trace",
}


# =============================================================================
# Environment Variables
# =============================================================================

ENV_RLM_MODE = "RLM_MODE"
ENV_RLM_RECURSION_DEPTH = "RLM_RECURSION_DEPTH"
ENV_RLM_MAX_DEPTH = "RLM_MAX_DEPTH"
ENV_CTX_FILE = "CTX_FILE"
ENV_RLM_STATE_FILE = "RLM_STATE_FILE"


# =============================================================================
# Pydantic Configuration Model
# =============================================================================

class RLMConfig(BaseModel):
    """Configuration model for RLM system with validation."""

    max_file_size_bytes: int = Field(
        default=MAX_FILE_SIZE_BYTES,
        ge=1024,  # At least 1KB
        description="Maximum file size in bytes before RLM tools are required"
    )

    max_chunk_size_tokens: int = Field(
        default=MAX_CHUNK_SIZE_TOKENS,
        ge=100,
        le=50000,
        description="Maximum tokens per chunk"
    )

    max_recursion_depth: int = Field(
        default=MAX_RECURSION_DEPTH,
        ge=1,
        le=10,
        description="Maximum recursion depth for sub-agents"
    )

    max_token_budget: int = Field(
        default=MAX_TOKEN_BUDGET,
        ge=1000,
        description="Maximum token budget per session"
    )

    cache_dir: str = Field(
        default=CACHE_DIR,
        description="Directory for temporary chunk files"
    )

    results_dir: str = Field(
        default=RESULTS_DIR,
        description="Directory for sub-agent results"
    )

    default_model: str = Field(
        default=DEFAULT_SUBAGENT_MODEL,
        description="Default model for sub-agents"
    )

    chunk_overlap_lines: int = Field(
        default=CHUNK_OVERLAP_LINES,
        ge=0,
        le=100,
        description="Number of overlapping lines between chunks"
    )

    workspace_root: Optional[Path] = Field(
        default=None,
        description="Root directory of the workspace"
    )

    model_config = ConfigDict(extra="forbid")  # Prevent unknown fields

    def get_cache_path(self) -> Path:
        """Get the absolute path to the cache directory."""
        root = self.workspace_root or Path.cwd()
        return root / self.cache_dir

    def get_results_path(self) -> Path:
        """Get the absolute path to the results directory."""
        root = self.workspace_root or Path.cwd()
        return root / self.results_dir

    def ensure_directories(self) -> None:
        """Create cache and results directories if they don't exist."""
        self.get_cache_path().mkdir(parents=True, exist_ok=True)
        self.get_results_path().mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def get_current_recursion_depth() -> int:
    """Get the current recursion depth from environment."""
    return int(os.environ.get(ENV_RLM_RECURSION_DEPTH, "0"))


def set_recursion_depth(depth: int) -> None:
    """Set the current recursion depth in environment."""
    os.environ[ENV_RLM_RECURSION_DEPTH] = str(depth)


def is_rlm_mode_active() -> bool:
    """Check if RLM mode is currently active."""
    return os.environ.get(ENV_RLM_MODE, "").lower() == "active"


def get_context_file() -> Optional[str]:
    """Get the current context file path from environment."""
    return os.environ.get(ENV_CTX_FILE)


def get_state_file() -> Path:
    """Get the path to the RLM state file."""
    state_file = os.environ.get(ENV_RLM_STATE_FILE)
    if state_file:
        return Path(state_file)
    return Path.cwd() / ".cache" / "rlm_state.json"


# =============================================================================
# Default Configuration Instance
# =============================================================================

def load_config(workspace_root: Optional[Path] = None) -> RLMConfig:
    """Load RLM configuration with optional workspace root."""
    config = RLMConfig(workspace_root=workspace_root)
    config.ensure_directories()
    return config
