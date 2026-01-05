"""Unit tests for the Knowledge Graph module."""

import pytest
import tempfile
import shutil
from pathlib import Path

from rlm_lib.knowledge_graph import (
    ProjectKnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)


@pytest.fixture
def temp_project():
    """Create a temporary project structure."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create Python files
    (temp_dir / "main.py").write_text('''
from utils import helper_function
from models import User

def main():
    user = User("test")
    result = helper_function(user)
    return result
''')
    
    (temp_dir / "utils.py").write_text('''
def helper_function(obj):
    return str(obj)

def another_helper():
    pass
''')
    
    (temp_dir / "models.py").write_text('''
class User:
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return f"User({self.name})"

class Admin(User):
    def __init__(self, name: str, role: str):
        super().__init__(name)
        self.role = role
''')
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def knowledge_graph(temp_project):
    """Create a knowledge graph for the temp project."""
    graph = ProjectKnowledgeGraph(temp_project)
    graph.build_index()
    return graph


class TestProjectKnowledgeGraph:
    def test_build_index(self, temp_project):
        graph = ProjectKnowledgeGraph(temp_project)
        stats = graph.build_index()
        assert stats["files_indexed"] >= 3
        assert stats["nodes_created"] > 0

    def test_find_by_name(self, knowledge_graph):
        nodes = knowledge_graph.find_by_name("User")
        assert len(nodes) >= 1
        assert any(n.name == "User" for n in nodes)

    def test_find_by_type(self, knowledge_graph):
        classes = knowledge_graph.find_by_type(NodeType.CLASS)
        assert len(classes) >= 2  # User and Admin

    def test_search(self, knowledge_graph):
        result = knowledge_graph.search("helper")
        assert len(result.nodes) >= 1

    def test_get_file_entities(self, knowledge_graph, temp_project):
        entities = knowledge_graph.get_file_entities(temp_project / "models.py")
        assert len(entities) >= 2  # User and Admin classes

    def test_get_stats(self, knowledge_graph):
        stats = knowledge_graph.get_stats()
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert stats["total_nodes"] > 0

    def test_save_and_load(self, knowledge_graph, temp_project):
        save_path = temp_project / "graph.json"
        knowledge_graph.save(save_path)
        assert save_path.exists()
        
        new_graph = ProjectKnowledgeGraph(temp_project)
        new_graph.load(save_path)
        assert new_graph.get_stats()["total_nodes"] == knowledge_graph.get_stats()["total_nodes"]


class TestGraphNode:
    def test_node_creation(self):
        node = GraphNode(
            node_id="test-id",
            name="TestClass",
            qualified_name="module.TestClass",
            node_type=NodeType.CLASS,
            file_path=Path("test.py"),
            line_start=1,
            line_end=10,
        )
        assert node.name == "TestClass"
        assert node.node_type == NodeType.CLASS


class TestGraphEdge:
    def test_edge_creation(self):
        edge = GraphEdge(
            source_id="node1",
            target_id="node2",
            edge_type=EdgeType.IMPORTS,
        )
        assert edge.source_id == "node1"
        assert edge.edge_type == EdgeType.IMPORTS


class TestNodeType:
    def test_node_types_exist(self):
        assert NodeType.FILE is not None
        assert NodeType.CLASS is not None
        assert NodeType.FUNCTION is not None
        assert NodeType.METHOD is not None


class TestEdgeType:
    def test_edge_types_exist(self):
        assert EdgeType.IMPORTS is not None
        assert EdgeType.CONTAINS is not None
        assert EdgeType.EXTENDS is not None
        assert EdgeType.CALLS is not None

