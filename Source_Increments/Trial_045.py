# Version 45

import sys
import re
from collections import defaultdict

class ControlFlowGraph:
    def __init__(self):
        # Initialize the graph as a dictionary of adjacency lists
        self.graph = defaultdict(list)
        self.edges = set()  # Store unique edges
        self.nodes = set()  # Store unique nodes
        self.paths = []  # List to store all found paths

    def add_edge(self, u, v):
        # Ensure node names are stripped of extra whitespace
        u, v = u.strip(), v.strip()
        
        # Detect control characters in node names
        if any(ord(c) < 32 or ord(c) == 127 for c in u+v):
            print(f"Error: Invalid node name detected: '{u}' or '{v}' contains control characters.")
            sys.exit(1)
        
        # Validate node names (must be alphanumeric, underscores, or hyphens)
        if not re.match(r'^[\w-]+$', u) or not re.match(r'^[\w-]+$', v):
            print(f"Error: Invalid node name: '{u}' or '{v}' (contains spaces or special characters)")
            sys.exit(1)
        
        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))  # Store edge for complexity calculation
        self.nodes.add(u)
        self.nodes.add(v)

    def find_basis_set(self, start, end):
        # Perform depth-first search (DFS) to find all paths from start to end
        path = []
        self.paths.clear()
        self._dfs(start, end, path)
        
        if not self.paths:
            print("Error: No paths found from start to end.")
            sys.exit(1)
        
        return self._extract_basis_set()

    def _dfs(self, node, end, path):
        path.append(node)  # Add current node to path
        
        if node == end:
            self.paths.append(list(path))  # Store a copy of the path
        else:
            for neighbor in self.graph[node]:
                if path.count(neighbor) < 2:  # Allow revisiting nodes, but limit cycles
                    self._dfs(neighbor, end, path)
        
        path.pop()  # Remove the last node before backtracking

    def _extract_basis_set(self):
        # Compute cyclomatic complexity with correct node counting
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        expected_basis_size = num_edges - num_nodes + 2
        
        # Ensure we collect the correct number of independent paths
        basis_set = []
        covered_edges = set()
        
        for path in self.paths:
            path_edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
            
            # Always add paths that introduce new edges
            if not path_edges.issubset(covered_edges):
                basis_set.append(path)
                covered_edges.update(path_edges)
            
            # Ensure we collect enough paths to match cyclomatic complexity
            if len(basis_set) >= expected_basis_size:
                break
        
        # If not enough paths have been found, keep adding remaining unique paths
        if len(basis_set) < expected_basis_size:
            remaining_paths = [p for p in self.paths if p not in basis_set]
            for path in remaining_paths:
                basis_set.append(path)
                if len(basis_set) >= expected_basis_size:
                    break
        
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.basis_path_count = len(basis_set)
        self.cyclomatic_complexity = expected_basis_size
        return basis_set

    @staticmethod
    def from_file(filename):
        # Check if file exists
        try:
            with open(filename, 'r') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        
        # Check if file is empty or contains only whitespace before processing
        if not lines or all(line.strip() == "" for line in lines):
            print("Error: Input file is empty or contains only whitespace.")
            sys.exit(1)
        
        # Check if the file has at least one edge defined before processing
        if len(lines[1:-1]) == 0:
            print("Error: No edges defined in the input file.")
            sys.exit(1)
        
                
        # Check for edges with too few or too many nodes before processing
        for line in lines[1:-1]:
            parts = line.split()
            if len(parts) < 2:
                print(f"Error: Edge definition '{line}' is incomplete. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
            if len(parts) > 2:
                print(f"Error: Edge definition '{line}' has too many nodes. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
        

        # Check for duplicate edges before processing
        seen_edges = set()
        for line in lines[1:-1]:
            parts = line.split()
            u, v = map(str.strip, parts)
            if (u, v) in seen_edges:
                print(f"Error: Duplicate edge detected: '{u} -> {v}'")
                sys.exit(1)
            seen_edges.add((u, v))
        
      
      # Validate file format
        if len(lines) < 3:
            print("Error: Invalid input file format. Expected at least a start node, edges, and an end node.")
            sys.exit(1)
        
        start = lines[0].strip()  # First line is the start node
        end = lines[-1].strip()  # Last line is the end node
        cfg = ControlFlowGraph()
        
        for line in lines[1:-1]:  # Process the edges in between
            parts = line.split()
            u, v = map(str.strip, parts)  # Strip whitespace
            cfg.add_edge(u, v)  # Add edge to graph
        
        # Check if start and end nodes are in the edges
        if start not in cfg.nodes:
            print(f"Error: STARTING NODE '{start}' is missing.")
            sys.exit(1)
        if end not in cfg.nodes:
            print(f"Error: ENDING NODE '{end}' is missing.")
            sys.exit(1)
        
        return cfg, start, end

# Main execution block
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]  # Get filename from command-line argument
    cfg, start, end = ControlFlowGraph.from_file(filename)  # Parse file
    
    basis_set = cfg.find_basis_set(start, end)  # Find basis set of paths
    
    # Print the discovered basis set of independent paths
    print("Basis Set of Paths:")
    for path in basis_set:
        print(" -> ".join(path))
    
    # Print the number of basis paths, nodes, edges, and computed cyclomatic complexity
    print(f"Number of Nodes: {cfg.num_nodes}")
    print(f"Number of Edges: {cfg.num_edges}")
    print(f"Cyclomatic Complexity (Expected Basis Paths): {cfg.cyclomatic_complexity}")
    print(f"Number of Basis Paths Found: {cfg.basis_path_count}")
