import sys
from collections import defaultdict

class ControlFlowGraph:
    def __init__(self):
        # Initialize the graph as a dictionary of adjacency lists
        self.graph = defaultdict(list)
        self.edges = set()  # Store unique edges
        self.nodes = set()  # Store unique nodes
        self.paths = []  # List to store all found paths
        self.back_edges = set()  # Track back edges (cycles)

    def add_edge(self, u, v):
        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))  # Store edge for complexity calculation
        self.nodes.add(u)
        self.nodes.add(v)

    def find_basis_set(self, start, end):
        # Perform depth-first search (DFS) to find all paths from start to end
        path = []
        self.paths.clear()
        self._dfs(start, end, set(), path, set())
        return self._extract_basis_set()

    def _dfs(self, node, end, visited_nodes, path, visited_edges):
        if node in visited_nodes:
            self.back_edges.add(node)  # Detect cycle
            return  # Stop recursion if cycle is detected
        
        path.append(node)  # Add current node to path
        visited_nodes.add(node)
        
        if node == end:
            self.paths.append(list(path))  # Store a copy of the path
        else:
            for neighbor in self.graph[node]:
                edge = (node, neighbor)
                if edge not in visited_edges or neighbor in self.back_edges:
                    visited_edges.add(edge)
                    self._dfs(neighbor, end, visited_nodes, path, visited_edges)
                    visited_edges.remove(edge)
        
        visited_nodes.remove(node)  # Allow revisiting nodes in different paths
        path.pop()  # Remove the last node before backtracking

    def _extract_basis_set(self):
        # Compute cyclomatic complexity with correct node counting
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        expected_basis_size = num_edges - num_nodes + 2
        
        print("All Paths Found:")
        for path in self.paths:
            print(" -> ".join(path))
        
        # Ensure we collect the correct number of independent paths
        basis_set = []
        covered_edges = set()
        
        for path in self.paths:
            path_edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
            
            # Always add paths that introduce new edges
            if not path_edges.issubset(covered_edges) or any(node in self.back_edges for node in path):
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
        # Read the control flow graph from a file
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
            start = lines[0]  # First line is the start node
            end = lines[-1]  # Last line is the end node
            cfg = ControlFlowGraph()
            for line in lines[1:-1]:  # Process the edges in between
                u, v = line.split()
                cfg.add_edge(u, v)  # Add edge to graph
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
