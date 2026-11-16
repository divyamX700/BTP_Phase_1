import os
import subprocess
import networkx as nx
import re

TGFF_BIN = os.path.join(os.path.dirname(__file__), '..', 'tgff-3.5', 'tgff')

def generate_tgff_dag(seed: int, num_tasks: int, num_machines: int, out_prefix="test"):
    """
    Generates a synthetic DAG using TGFF and parses it into a NetworkX DiGraph.
    Each node will have a 'p_ij' attribute: a list of execution times for each machine.
    """
    opt_file = f"{out_prefix}.tgffopt"
    
    # 1. Write the .tgffopt configuration file
    with open(opt_file, "w", newline='\n') as f:
        f.write(f"seed {seed}\n")
        f.write("tg_cnt 1\n")
        f.write(f"task_cnt {num_tasks} 1\n") # average tasks, multiplier
        f.write("task_degree 2 2\n") # max in/out degree
        f.write("period_laxity 1\n")
        f.write("period_mul 1, 2\n")
        f.write("tg_write\n")
        
        # Processor tables (unrelated machines)
        f.write("table_label PROCESSOR\n")
        f.write(f"table_cnt {num_machines}\n")
        f.write("table_attrib price 80 20\n")
        f.write("type_attrib exec_time 50 40\n") 
        f.write("pe_write\n")

    # 2. Execute TGFF (assuming wsl is required on Windows)
    # the argument to tgff is the prefix, it automatically appends .tgffopt
    cmd = ["wsl", "-d", "Ubuntu", "-e", "./tgff-3.5/tgff", out_prefix]
    subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'), check=True, capture_output=True)

    # 3. Parse the generated .tgff file
    tgff_out = f"{out_prefix}.tgff"
    
    G = nx.DiGraph()
    
    current_section = None
    current_processor = -1
    
    with open(tgff_out, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            # Detect sections
            if line.startswith("@TASK_GRAPH"):
                current_section = "TASK_GRAPH"
                continue
            elif line.startswith("@PROCESSOR"):
                current_section = "PROCESSOR"
                match = re.match(r"@PROCESSOR\s+(\d+)", line)
                if match:
                    current_processor = int(match.group(1))
                continue
                
            # Parse TASK_GRAPH
            if current_section == "TASK_GRAPH":
                if line.startswith("TASK"):
                    # TASK <id> TYPE <type_id>
                    parts = line.split()
                    task_id = parts[1]
                    task_type = parts[3]
                    G.add_node(task_id, type=task_type, p_ij=[0]*num_machines)
                elif line.startswith("ARC"):
                    # ARC <id> FROM <node> TO <node> TYPE <type_id>
                    parts = line.split()
                    u = parts[3]
                    v = parts[5]
                    G.add_edge(u, v)
                    
            # Parse PROCESSOR tables
            elif current_section == "PROCESSOR":
                # Data lines look like: <type_id> <exec_time>
                # e.g., 0 45.2
                if line == "}":
                    current_section = None
                    continue
                
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    task_type = parts[0]
                    exec_time = float(parts[2])
                    
                    # Update all nodes of this type with their execution time on this processor
                    for node, data in G.nodes(data=True):
                        if data.get('type') == task_type:
                            data['p_ij'][current_processor] = exec_time

    return G

if __name__ == "__main__":
    # Quick test
    dag = generate_tgff_dag(seed=42, num_tasks=10, num_machines=3, out_prefix="data/sample")
    print(f"Generated DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges.")
    for node, data in dag.nodes(data=True):
        print(f"Task {node} (Type {data['type']}): p_ij = {data['p_ij']}")
