"""
RLM-C Enterprise Context Cache Module

Provides persistent context storage with intelligent invalidation for
maintaining cross-session awareness of large codebases.

This module is part of the Enterprise RLM-CLAUDE system designed for
large-scale codebase management (10,000+ files, multi-million LOC).
"""

import os
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time

from .config import RLMConfig, load_config


class CacheEntryType(Enum):
    """Types of cached entries."""
    FILE_CONTENT = "file_content"
    FILE_ANALYSIS = "file_analysis"
    CHUNK = "chunk"
    QUERY_RESULT = "query_result"
    SYMBOL_INFO = "symbol_info"
    DEPENDENCY_MAP = "dependency_map"
    CONTEXT_SUMMARY = "context_summary"


@dataclass
class CacheEntry:
    """A cached entry with metadata."""
    key: str
    entry_type: CacheEntryType
    content: str
    source_path: Optional[str] = None
    source_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "entry_type": self.entry_type.value,
            "content": self.content,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            entry_type=CacheEntryType(data["entry_type"]),
            content=data["content"],
            source_path=data.get("source_path"),
            source_hash=data.get("source_hash"),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            ttl_seconds=data.get("ttl_seconds"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheStats:
    """Statistics about cache usage."""
    total_entries: int
    total_size_bytes: int
    hits: int
    misses: int
    evictions: int
    entries_by_type: Dict[str, int]

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ContextCache:
    """
    Enterprise-grade context cache for persistent storage.

    Features:
    - SQLite-backed persistence
    - Automatic invalidation based on file changes
    - LRU eviction policy
    - TTL support for time-sensitive entries
    - Thread-safe operations
    """

    def __init__(
        self,
        config: Optional[RLMConfig] = None,
        max_size_mb: int = 500,
        default_ttl_seconds: Optional[int] = None,
    ):
        """Initialize the context cache."""
        self.config = config or load_config()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._db_path = self.config.get_cache_path() / "context_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self.config.ensure_directories()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_path TEXT,
                    source_hash TEXT,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_path ON cache_entries(source_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_type ON cache_entries(entry_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
            conn.commit()

    # =========================================================================
    # Core Cache Operations
    # =========================================================================

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get an entry from the cache."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if row is None:
                    self._misses += 1
                    return None

                entry = self._row_to_entry(row)
                if entry.is_expired():
                    self.delete(key)
                    self._misses += 1
                    return None

                # Update access stats
                conn.execute(
                    """UPDATE cache_entries
                       SET accessed_at = ?, access_count = access_count + 1
                       WHERE key = ?""",
                    (datetime.now().isoformat(), key)
                )
                conn.commit()
                self._hits += 1
                return entry

    def set(
        self,
        key: str,
        content: str,
        entry_type: CacheEntryType,
        source_path: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """Set an entry in the cache."""
        with self._lock:
            # Calculate source hash if path provided
            source_hash = None
            if source_path and Path(source_path).exists():
                source_hash = self._compute_file_hash(Path(source_path))

            entry = CacheEntry(
                key=key,
                entry_type=entry_type,
                content=content,
                source_path=source_path,
                source_hash=source_hash,
                ttl_seconds=ttl_seconds or self.default_ttl,
                metadata=metadata or {},
            )

            # Check size and evict if necessary
            self._ensure_capacity(len(content.encode('utf-8')))

            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cache_entries
                       (key, entry_type, content, source_path, source_hash,
                        created_at, accessed_at, access_count, ttl_seconds, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.key,
                        entry.entry_type.value,
                        entry.content,
                        entry.source_path,
                        entry.source_hash,
                        entry.created_at.isoformat(),
                        entry.accessed_at.isoformat(),
                        entry.access_count,
                        entry.ttl_seconds,
                        json.dumps(entry.metadata),
                    )
                )
                conn.commit()
            return entry

    def delete(self, key: str) -> bool:
        """Delete an entry from the cache."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0

    def invalidate_by_source(self, source_path: str) -> int:
        """Invalidate all entries associated with a source file."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE source_path = ?", (source_path,)
                )
                conn.commit()
                count = cursor.rowcount
                self._evictions += count
                return count

    def invalidate_stale(self) -> int:
        """Invalidate entries whose source files have changed."""
        with self._lock:
            stale_keys: List[str] = []
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT key, source_path, source_hash FROM cache_entries WHERE source_path IS NOT NULL"
                )
                for row in cursor:
                    source_path = Path(row["source_path"])
                    if source_path.exists():
                        current_hash = self._compute_file_hash(source_path)
                        if current_hash != row["source_hash"]:
                            stale_keys.append(row["key"])
                    else:
                        stale_keys.append(row["key"])

            for key in stale_keys:
                self.delete(key)
            self._evictions += len(stale_keys)
            return len(stale_keys)

    def clear(self) -> int:
        """Clear all entries from the cache."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache_entries")
                conn.commit()
                count = cursor.rowcount
                self._evictions += count
                return count

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        """Convert a database row to a CacheEntry."""
        return CacheEntry(
            key=row["key"],
            entry_type=CacheEntryType(row["entry_type"]),
            content=row["content"],
            source_path=row["source_path"],
            source_hash=row["source_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            access_count=row["access_count"],
            ttl_seconds=row["ttl_seconds"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _compute_file_hash(self, path: Path) -> str:
        """Compute a hash of a file's content and modification time."""
        stat = path.stat()
        hash_input = f"{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure there's enough capacity, evicting if necessary."""
        current_size = self._get_total_size()
        while current_size + needed_bytes > self.max_size_bytes:
            evicted = self._evict_lru()
            if not evicted:
                break
            current_size = self._get_total_size()

    def _get_total_size(self) -> int:
        """Get total size of cached content in bytes."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("SELECT SUM(LENGTH(content)) FROM cache_entries")
            result = cursor.fetchone()[0]
            return result or 0

    def _evict_lru(self) -> bool:
        """Evict the least recently used entry."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "SELECT key FROM cache_entries ORDER BY accessed_at ASC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (row[0],))
                conn.commit()
                self._evictions += 1
                return True
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]

                cursor = conn.execute("SELECT entry_type, COUNT(*) FROM cache_entries GROUP BY entry_type")
                entries_by_type = {row[0]: row[1] for row in cursor}

            return CacheStats(
                total_entries=total_entries,
                total_size_bytes=self._get_total_size(),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries_by_type=entries_by_type,
            )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def cache_file_analysis(self, file_path: Path, analysis: Dict[str, Any]) -> CacheEntry:
        """Cache a file analysis result."""
        key = f"analysis::{file_path}"
        return self.set(
            key=key,
            content=json.dumps(analysis),
            entry_type=CacheEntryType.FILE_ANALYSIS,
            source_path=str(file_path),
        )

    def get_file_analysis(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached file analysis if valid."""
        key = f"analysis::{file_path}"
        entry = self.get(key)
        if entry:
            return json.loads(entry.content)
        return None

    def cache_query_result(self, query: str, result: str, ttl_seconds: int = 3600) -> CacheEntry:
        """Cache a query result with TTL."""
        key = f"query::{hashlib.md5(query.encode()).hexdigest()}"
        return self.set(
            key=key,
            content=result,
            entry_type=CacheEntryType.QUERY_RESULT,
            ttl_seconds=ttl_seconds,
            metadata={"query": query},
        )

    def get_query_result(self, query: str) -> Optional[str]:
        """Get cached query result if valid."""
        key = f"query::{hashlib.md5(query.encode()).hexdigest()}"
        entry = self.get(key)
        return entry.content if entry else None
