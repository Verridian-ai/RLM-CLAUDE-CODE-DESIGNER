#!/usr/bin/env python3
"""
Test RLM System with PDF File

This script tests the RLM-CLAUDE system using the provided PDF file as a test case.
It validates all core RLM functionality including previewing, chunking, and processing.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add the rlm_lib to path
sys.path.insert(0, str(Path(__file__).parent))

from rlm_lib import (
    RLMKernel,
    RLMConfig,
    get_file_info,
    preview_file,
    chunk_data,
)
from rlm_lib.kernel import process_query

# Test file path
PDF_FILE = Path(__file__).parent / "Claude Code RLM Implementation Research.pdf"

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

def test_file_exists():
    """Test that the PDF file exists and get basic info."""
    print_section("FILE EXISTENCE AND BASIC INFO")

    if not PDF_FILE.exists():
        raise FileNotFoundError(f"Test PDF file not found: {PDF_FILE}")

    print(f"[OK] PDF file exists: {PDF_FILE}")
    print(f"[OK] File size: {PDF_FILE.stat().st_size:,} bytes")

    return True

def test_get_file_info():
    """Test getting file information using RLM tools."""
    print_section("RLM FILE INFO")

    try:
        info = get_file_info(PDF_FILE)
        print(f"[OK] File path: {info.path}")
        print(f"[OK] Size: {info.size_human} ({info.size_bytes:,} bytes)")
        print(f"[OK] Content type: {info.content_type}")
        print(f"[OK] Extension: {info.extension}")
        print(f"[OK] Modified: {info.modified_time}")
        print(f"[OK] Is binary: {info.is_binary}")
        print(f"[OK] Encoding: {info.encoding}")

        # Check if file requires RLM
        kernel = RLMKernel()
        requires_rlm = kernel.requires_rlm(PDF_FILE)
        print(f"[OK] Requires RLM processing: {requires_rlm}")

        return info, requires_rlm

    except Exception as e:
        print(f"[FAIL] Error getting file info: {e}")
        traceback.print_exc()
        return None, None

def test_preview_functionality():
    """Test RLM preview functionality."""
    print_section("RLM PREVIEW FUNCTIONALITY")

    try:
        # Test preview
        preview = preview_file(PDF_FILE, lines=10)
        print(f"[OK] Preview generated ({len(preview)} characters)")
        print(f"Preview content (first 200 chars):\n{preview[:200]}...")

        if len(preview) > 0:
            print("[OK] Preview contains content")
        else:
            print("[FAIL] Preview is empty")

        return preview

    except Exception as e:
        print(f"[FAIL] Error in preview: {e}")
        traceback.print_exc()
        return None

def test_chunking():
    """Test RLM chunking functionality."""
    print_section("RLM CHUNKING FUNCTIONALITY")

    try:
        # Initialize config
        config = RLMConfig()
        config.ensure_directories()

        # Test chunking
        print(f"Chunking file: {PDF_FILE}")
        chunks = chunk_data(PDF_FILE, config=config)

        print(f"[OK] Created {len(chunks)} chunks")

        # Examine first chunk
        if chunks:
            first_chunk = chunks[0]
            print(f"[OK] First chunk: {first_chunk.chunk_path}")
            print(f"[OK] Lines: {first_chunk.start_line}-{first_chunk.end_line}")
            print(f"[OK] Size: {first_chunk.size_bytes:,} bytes")
            print(f"[OK] Content type: {first_chunk.content_type}")

            # Verify chunk file exists
            if first_chunk.chunk_path.exists():
                print("[OK] Chunk file exists on disk")

                # Read a small portion of the chunk
                with open(first_chunk.chunk_path, 'r', encoding='utf-8') as f:
                    chunk_content = f.read(200)
                print(f"[OK] Chunk content preview: {chunk_content[:100]}...")
            else:
                print("[FAIL] Chunk file does not exist")

        return chunks

    except Exception as e:
        print(f"[FAIL] Error in chunking: {e}")
        traceback.print_exc()
        return None

def test_kernel_initialization():
    """Test RLM kernel initialization."""
    print_section("RLM KERNEL INITIALIZATION")

    try:
        # Initialize kernel
        kernel = RLMKernel()
        kernel.initialize(context_file=str(PDF_FILE))

        print("[OK] Kernel initialized successfully")

        # Get status
        status = kernel.get_status()
        print(f"[OK] Kernel status: {status['kernel_status']}")
        print(f"[OK] Session ID: {status['session_id']}")
        print(f"[OK] RLM mode active: {status['rlm_mode_active']}")
        print(f"[OK] Context file: {status['context_file']}")

        return kernel

    except Exception as e:
        print(f"[FAIL] Error initializing kernel: {e}")
        traceback.print_exc()
        return None

def test_simple_query():
    """Test a simple query against the PDF."""
    print_section("RLM QUERY PROCESSING")

    try:
        # Simple query
        query = "What is this document about? Provide a brief summary."

        print(f"Testing query: {query}")

        # Use process_query function
        result = process_query(
            query=query,
            context_file=str(PDF_FILE),
            model="haiku"  # Use faster model for testing
        )

        print(f"[OK] Query processed successfully")
        print(f"[OK] Total tasks: {result.total_tasks}")
        print(f"[OK] Successful tasks: {result.successful_tasks}")
        print(f"[OK] Failed tasks: {result.failed_tasks}")
        print(f"[OK] Total tokens: {result.total_tokens}")
        print(f"[OK] Duration: {result.total_duration:.2f}s")

        # Show first result
        if result.results:
            first_result = result.results[0]
            print(f"[OK] First result status: {first_result.status}")
            if first_result.output:
                print(f"[OK] Output preview: {first_result.output[:200]}...")

        return result

    except Exception as e:
        print(f"[FAIL] Error in query processing: {e}")
        traceback.print_exc()
        return None

def run_all_tests():
    """Run all RLM tests with the PDF file."""
    print_section("RLM-CLAUDE SYSTEM TEST WITH PDF")
    print(f"Test file: {PDF_FILE}")
    print(f"Test time: {datetime.now()}")

    test_results = {}

    try:
        # Test 1: File existence
        test_results['file_exists'] = test_file_exists()

        # Test 2: File info
        info, requires_rlm = test_get_file_info()
        test_results['file_info'] = info is not None
        test_results['requires_rlm'] = requires_rlm

        # Test 3: Preview
        preview = test_preview_functionality()
        test_results['preview'] = preview is not None

        # Test 4: Chunking
        chunks = test_chunking()
        test_results['chunking'] = chunks is not None and len(chunks) > 0

        # Test 5: Kernel initialization
        kernel = test_kernel_initialization()
        test_results['kernel_init'] = kernel is not None

        # Test 6: Simple query (if file is large enough)
        if requires_rlm:
            result = test_simple_query()
            test_results['query_processing'] = result is not None
        else:
            print("\n[WARN]  File doesn't require RLM (too small), skipping query test")
            test_results['query_processing'] = None

    except Exception as e:
        print(f"\n[FAIL] Fatal error during testing: {e}")
        traceback.print_exc()

    # Summary
    print_section("TEST SUMMARY")
    passed = 0
    total = 0

    for test_name, result in test_results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
            passed += 1
        else:
            status = "FAILED"

        if result is not None:
            total += 1

        print(f"{test_name:20}: {status}")

    print(f"\nOVERALL: {passed}/{total} tests passed")

    if passed == total and total > 0:
        print("[SUCCESS] All tests PASSED!")
        return True
    else:
        print("[ERROR] Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)