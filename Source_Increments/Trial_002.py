import sys
from collections import defaultdict

class ControlFlowGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.paths = []

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def find_basis_set(self, start, end):
        visited = set()
        path = []
        self._dfs(start, end, visited, path)
        return self.paths

    def _dfs(self, node, end, visited, path):
        path.append(node)
        if node == end:
            self.paths.append(list(path))  # Store a copy of the path
        else:
            visited.add(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited or neighbor == end:
                    self._dfs(neighbor, end, visited, path)
            visited.remove(node)  # Backtrack
        path.pop()

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
            start = lines[0]
            end = lines[-1]
            cfg = ControlFlowGraph()
            for line in lines[1:-1]:
                u, v = line.split()
                cfg.add_edge(u, v)
            return cfg, start, end

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    cfg, start, end = ControlFlowGraph.from_file(filename)
    
    print(start)
    print(end)
    for nodes in cfg.graph:
        print(nodes)
        
    for edges in cfg.graph:
        print(edges)
    
    basis_set = cfg.find_basis_set(start, end)
    print("Basis Set of Paths:")
    for path in basis_set:
        print(" -> ".join(path))
