import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np

def parse_pegasus_dax(xml_path: str, num_machines: int, seed: int = 42) -> nx.DiGraph:
    """
    Parses a Pegasus DAX XML file and converts it into a NetworkX DiGraph.
    Synthesizes the p_ij execution matrix based on the base 'runtime' attribute.
    """
    np.random.seed(seed)
    
    # Remove XML namespaces if they exist to simplify finding tags
    it = ET.iterparse(xml_path)
    for _, el in it:
        _, _, el.tag = el.tag.rpartition('}') 
    root = it.root

    G = nx.DiGraph()

    # 1. Parse Jobs (Nodes)
    for job in root.findall("job"):
        job_id = job.get("id")
        base_runtime = float(job.get("runtime", 1.0))
        
        # Synthesize heterogeneity across m machines.
        # We assume machines can be up to 2x faster or 2x slower than the nominal runtime.
        heterogeneity_factors = np.random.uniform(0.5, 2.0, size=num_machines)
        p_ij = (base_runtime * heterogeneity_factors).tolist()
        
        G.add_node(job_id, runtime=base_runtime, p_ij=p_ij)

    # 2. Parse Dependencies (Edges)
    for child in root.findall("child"):
        child_id = child.get("ref")
        for parent in child.findall("parent"):
            parent_id = parent.get("ref")
            G.add_edge(parent_id, child_id)

    # 3. Handle missing nodes (some Pegasus XMLs have broken edge references)
    for node, data in list(G.nodes(data=True)):
        if "p_ij" not in data:
            data["runtime"] = 0.0
            data["p_ij"] = [0.0] * num_machines

    return G

if __name__ == "__main__":
    # Quick test
    import sys
    import os
    
    sample_file = os.path.join(os.path.dirname(__file__), "..", "WF", "WorkFlowBenchmark", "Montage", "Montage_25.xml")
    if os.path.exists(sample_file):
        dag = parse_pegasus_dax(sample_file, num_machines=4)
        print(f"Parsed Pegasus DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
        # Print a sample node
        sample_node = list(dag.nodes)[0]
        print(f"Sample Node {sample_node} p_ij: {dag.nodes[sample_node]['p_ij']}")
    else:
        print(f"Could not find sample file at {sample_file}")
