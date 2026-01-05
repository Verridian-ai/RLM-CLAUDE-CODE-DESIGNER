"""Unit tests for the Semantic Chunker module."""

import pytest
import tempfile
from pathlib import Path

from rlm_lib.semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    SemanticBoundaryType,
    PythonSemanticStrategy,
    TypeScriptSemanticStrategy,
    semantic_chunk_file,
    analyze_file_semantics,
)


@pytest.fixture
def python_sample_code():
    return '''"""Sample module."""
import os
from pathlib import Path

class MyClass:
    """A sample class."""
    def __init__(self, name: str):
        self.name = name
    def greet(self) -> str:
        return f"Hello, {self.name}!"

def standalone_function(x: int, y: int) -> int:
    return x + y

async def async_function():
    pass
'''


@pytest.fixture
def typescript_sample_code():
    return '''import { Component } from 'react';
import axios from 'axios';

export interface UserProps {
    name: string;
    age: number;
}

export class UserComponent extends Component<UserProps> {
    render() {
        return <div>{this.props.name}</div>;
    }
}

export function calculateAge(birthYear: number): number {
    return new Date().getFullYear() - birthYear;
}
'''


@pytest.fixture
def temp_python_file(python_sample_code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_sample_code)
        return Path(f.name)


@pytest.fixture
def temp_typescript_file(typescript_sample_code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
        f.write(typescript_sample_code)
        return Path(f.name)


class TestPythonSemanticStrategy:
    def test_extract_boundaries_finds_classes(self, python_sample_code):
        strategy = PythonSemanticStrategy()
        boundaries = strategy.extract_boundaries(python_sample_code, Path("test.py"))
        class_boundaries = [b for b in boundaries if b.boundary_type == SemanticBoundaryType.CLASS]
        assert len(class_boundaries) == 1
        assert class_boundaries[0].name == "MyClass"

    def test_extract_boundaries_finds_functions(self, python_sample_code):
        strategy = PythonSemanticStrategy()
        boundaries = strategy.extract_boundaries(python_sample_code, Path("test.py"))
        func_boundaries = [b for b in boundaries if b.boundary_type == SemanticBoundaryType.FUNCTION]
        assert len(func_boundaries) >= 2

    def test_extract_imports(self, python_sample_code):
        strategy = PythonSemanticStrategy()
        imports = strategy.extract_imports(python_sample_code)
        assert "os" in imports

    def test_handles_syntax_errors_gracefully(self):
        strategy = PythonSemanticStrategy()
        invalid_code = "class Broken(\n    def method(self):"
        boundaries = strategy.extract_boundaries(invalid_code, Path("test.py"))
        assert isinstance(boundaries, list)


class TestTypeScriptSemanticStrategy:
    def test_extract_boundaries_finds_classes(self, typescript_sample_code):
        strategy = TypeScriptSemanticStrategy()
        boundaries = strategy.extract_boundaries(typescript_sample_code, Path("test.ts"))
        class_boundaries = [b for b in boundaries if b.boundary_type == SemanticBoundaryType.CLASS]
        assert len(class_boundaries) >= 1

    def test_extract_boundaries_finds_interfaces(self, typescript_sample_code):
        strategy = TypeScriptSemanticStrategy()
        boundaries = strategy.extract_boundaries(typescript_sample_code, Path("test.ts"))
        names = [b.name for b in boundaries]
        assert "UserProps" in names

    def test_extract_imports(self, typescript_sample_code):
        strategy = TypeScriptSemanticStrategy()
        imports = strategy.extract_imports(typescript_sample_code)
        assert "react" in imports
        assert "axios" in imports


class TestSemanticChunker:
    def test_chunk_python_file(self, temp_python_file):
        chunker = SemanticChunker()
        chunks = chunker.chunk_file(temp_python_file)
        assert len(chunks) >= 1
        assert all(isinstance(c, SemanticChunk) for c in chunks)
        temp_python_file.unlink()

    def test_chunk_typescript_file(self, temp_typescript_file):
        chunker = SemanticChunker()
        chunks = chunker.chunk_file(temp_typescript_file)
        assert len(chunks) >= 1
        temp_typescript_file.unlink()

    def test_analyze_file(self, temp_python_file):
        chunker = SemanticChunker()
        analysis = chunker.analyze_file(temp_python_file)
        assert analysis["semantic_support"] is True
        assert analysis["language"] == "python"
        temp_python_file.unlink()


class TestConvenienceFunctions:
    def test_semantic_chunk_file(self, temp_python_file):
        chunks = semantic_chunk_file(temp_python_file)
        assert len(chunks) >= 1
        temp_python_file.unlink()

    def test_analyze_file_semantics(self, temp_python_file):
        analysis = analyze_file_semantics(temp_python_file)
        assert "path" in analysis
        assert "semantic_support" in analysis
        temp_python_file.unlink()

