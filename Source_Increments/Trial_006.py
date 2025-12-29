import sys
from collections import defaultdict

class ControlFlowGraph:
    def __init__(self):
        # Initialize the graph as a dictionary of adjacency lists
        self.graph = defaultdict(list)
        self.edges = set()  # Store unique edges
        self.paths = []  # List to store all found paths
        self.visited_edges = set()  # Track visited edges to handle cycles

    def add_edge(self, u, v):
        # Add a directed edge from node u to node v
        self.graph[u].append(v)
        self.edges.add((u, v))  # Store edge for complexity calculation

    def find_basis_set(self, start, end):
        # Perform depth-first search (DFS) to find all paths from start to end
        visited_nodes = set()
        path = []
        self.visited_edges.clear()
        self._dfs(start, end, visited_nodes, path)
        return self._extract_basis_set()

    def _dfs(self, node, end, visited_nodes, path):
        # Recursive DFS function to explore paths
        path.append(node)  # Add current node to path
        
        if node == end:
            # If we reach the end node, store the found path
            self.paths.append(list(path))  # Store a copy of the path
        else:
            visited_nodes.add(node)  # Mark the node as visited
            for neighbor in self.graph[node]:
                edge = (node, neighbor)
                if edge not in self.visited_edges:  # Avoid redundant traversals
                    self.visited_edges.add(edge)
                    self._dfs(neighbor, end, visited_nodes, path)
            visited_nodes.remove(node)  # Backtrack to explore other paths
        
        path.pop()  # Remove the last node before backtracking

    def _extract_basis_set(self):
        # Compute cyclomatic complexity
        num_nodes = len(self.graph)
        num_edges = len(self.edges)
        self.cyclomatic_complexity = num_edges - num_nodes + 2
        
        # Ensure we only keep a linearly independent set of paths
        basis_set = []
        covered_edges = set()
        
        for path in self.paths:
            path_edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
            
            # A path is independent if it introduces a new edge
            if not path_edges.issubset(covered_edges):
                basis_set.append(path)
                covered_edges.update(path_edges)
                
            # Stop if we have enough paths to match cyclomatic complexity
            if len(basis_set) == self.cyclomatic_complexity:
                break
        
        self.basis_path_count = len(basis_set)  # Store number of basis paths
        self.num_nodes = num_nodes  # Store number of nodes
        self.num_edges = num_edges  # Store number of edges
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
