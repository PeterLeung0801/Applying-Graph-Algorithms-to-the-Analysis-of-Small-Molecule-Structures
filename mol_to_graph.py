from rdkit import Chem
from rdkit.Chem import rdDepictor
import networkx as nx
import pickle

smiles_list = [
    "CCN(CC)C(=O)c1cccnc1",
    "CCN1C(=O)NC(c2ccccc2)C1=O"
]

for i in range(len(smiles_list)):
    smiles = smiles_list[i]
    print(f"Processing molecule {i+1}... ({smiles})")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES!")
        continue
    
    rdDepictor.Compute2DCoords(mol)

    # Create two graphs
    G_unweighted = nx.Graph()   # For BFS and DFS (no weights)
    G_weighted = nx.Graph()     # For Kruskal and Dijkstra (with weights)
    
    # Add atoms as nodes
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        label = f"{symbol}{idx}"          # e.g. C7, N2, O6

        G_unweighted.add_node(label)
        G_weighted.add_node(label)
    
    # Add bonds as edges with weights (single=1, double=2)
    for bond in mol.GetBonds():
        a1_idx = bond.GetBeginAtomIdx()
        a2_idx = bond.GetEndAtomIdx()
        
        label1 = f"{mol.GetAtomWithIdx(a1_idx).GetSymbol()}{a1_idx}"
        label2 = f"{mol.GetAtomWithIdx(a2_idx).GetSymbol()}{a2_idx}"
        
        # Set weight: double bond = 2, others (single/aromatic) = 1
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            weight = 2
        else:
            weight = 1
            
        # Add to both graphs
        G_unweighted.add_edge(label1, label2)
        G_weighted.add_edge(label1, label2, weight=weight)
    
    # Print graph information
    print(f"Number of atoms (nodes): {G_unweighted.number_of_nodes()}")
    print(f"Number of bonds (edges): {G_unweighted.number_of_edges()}")
    
    # Check for atom with index 7 (C7)
    labels = sorted(G_unweighted.nodes())
    print(f"Nodes: {labels}")

    if "C7" in G_unweighted:
        print(f"Found atom C7, label is C7")
    else:
        print("This molecule does not have atom with label C7")

    if "C0" in G_unweighted:
        print(f"Found atom C0, label is C0")
    else:
        print("This molecule does not have atom with label C0")


    # Save unweighted graph
    filename_un = f"molecule_{i+1}_unweighted.pkl"
    with open(filename_un, "wb") as f:
        pickle.dump(G_unweighted, f)
    print("Saved:", filename_un)

    # Save weighted graph
    filename_w = f"molecule_{i+1}_weighted.pkl"
    with open(filename_w, "wb") as f:
        pickle.dump(G_weighted, f)
    print("Saved:", filename_w)

    print("---")

print("All molecules processed!")