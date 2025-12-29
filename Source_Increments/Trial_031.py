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
        
        # Detect duplicate edges
        if (u, v) in self.edges:
            print(f"Warning: Duplicate edge '{u} -> {v}' found and ignored.")
            return
        
        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))  # Store edge for complexity calculation
        self.nodes.add(u)
        self.nodes.add(v)

    def _is_reachable(self, start, target):
        """Check if 'target' node is reachable from 'start' using DFS."""
        visited = set()
        def dfs(node):
            if node == target:
                return True
            visited.add(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited and dfs(neighbor):
                    return True
            return False
        return dfs(start)

    def find_basis_set(self, start, end):
        # Ensure the START node can reach all other nodes
        for node in self.nodes:
            if node != start and not self._is_reachable(start, node):
                print(f"Error: Node '{node}' is disconnected from the START node '{start}'.")
                sys.exit(1)
        
        # Ensure the END node is reachable from at least one node
        if not any(self._is_reachable(node, end) for node in self.nodes if node != end):
            print(f"Error: The END node '{end}' is disconnected from the rest of the graph.")
            sys.exit(1)
        
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

    @staticmethod
    def from_file(filename):
        # Check if file exists
        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        
        # Validate file is not empty
        if not lines:
            print("Error: Input file is empty or contains only whitespace.")
            sys.exit(1)
        
        # Validate file format
        if len(lines) < 3:
            print("Error: Invalid input file format. Expected at least a start node, edges, and an end node.")
            sys.exit(1)
        
        start = lines[0]  # First line is the start node
        end = lines[-1]  # Last line is the end node
        cfg = ControlFlowGraph()
        
        for line in lines[1:-1]:  # Process the edges in between
            parts = line.split()
            if len(parts) != 2:
                print(f"Error: Invalid edge definition '{line}'. Expected format: 'NODE1 NODE2'")
                sys.exit(1)
            u, v = parts
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
