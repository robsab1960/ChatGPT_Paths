import sys
import random

def generate_graph(num_nodes, file_name):
    # Create a list of nodes
    nodes = [f"Node{i}" for i in range(1, num_nodes + 1)]
    
    # Randomly create edges between nodes
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.choice([True, False]):
                edges.append((nodes[i], nodes[j]))
    
    # Write the graph to the file
    with open(file_name, 'w') as f:
        f.write(f"{nodes[0]}\n")  # Start node
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
        f.write(f"{nodes[-1]}\n")  # End node

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <num_nodes> <file_name>")
        sys.exit(1)
    
    num_nodes = int(sys.argv[1])
    file_name = sys.argv[2]
    
    if num_nodes <= 0:
        print("Number of nodes must be a positive whole number.")
        sys.exit(1)
    
    generate_graph(num_nodes, file_name)