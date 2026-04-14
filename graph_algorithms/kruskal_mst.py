from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from PIL import ImageDraw
import networkx as nx
import pickle
import re

print("Starting Kruskal's algorithm (MST) for both molecules...\n")

smiles_list = [
    "CCN(CC)C(=O)c1cccnc1",      # Molecule 1
    "CCN1C(=O)NC(c2ccccc2)C1=O"  # Molecule 2
]


def atom_index_from_label(label: str) -> int:
    match = re.search(r"(\d+)$", label)
    if not match:
        raise ValueError(f"Cannot parse atom index from label: {label}")
    return int(match.group(1))


def node_sort_key(label: str):
    match = re.match(r"([A-Za-z]+)(\d+)$", label)
    if not match:
        return (label, -1)
    return (match.group(1), int(match.group(2)))


def union_find_init(nodes):
    parent = {n: n for n in nodes}
    rank = {n: 0 for n in nodes}
    return parent, rank


def union_find_find(parent, node):
    if parent[node] != node:
        parent[node] = union_find_find(parent, parent[node])
    return parent[node]


def union_find_union(parent, rank, u, v):
    root_u = union_find_find(parent, u)
    root_v = union_find_find(parent, v)
    if root_u == root_v:
        return False
    if rank[root_u] < rank[root_v]:
        parent[root_u] = root_v
    elif rank[root_u] > rank[root_v]:
        parent[root_v] = root_u
    else:
        parent[root_v] = root_u
        rank[root_u] += 1
    return True


def annotate_atom_labels(mol):
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetProp("atomNote", f"{atom.GetSymbol()}{idx}")


def format_edge_lines(edges, chunk_size: int = 4):
    edge_texts = [f"{u}-{v}(w={data['weight']})" for u, v, data in edges]
    chunks = [edge_texts[i:i + chunk_size] for i in range(0, len(edge_texts), chunk_size)]
    lines = []
    for i, chunk in enumerate(chunks):
        text = " | ".join(chunk)
        if i == 0:
            lines.append(f"MST edges: {text}")
        else:
            lines.append(f"           {text}")
    return lines


def add_overlay(img, lines):
    draw = ImageDraw.Draw(img)
    line_height = 18
    top = 6
    left = 6
    right = 794
    bottom = top + line_height * len(lines) + 8
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255))
    y = top + 4
    for line in lines:
        draw.text((left + 6, y), line, fill=(0, 0, 0))
        y += line_height


def smiles_to_filename(smiles: str) -> str:
    for ch in r'\/:*?"<>|':
        smiles = smiles.replace(ch, "_")
    return smiles


for mol_id in [1, 2]:
    print(f"=== Molecule {mol_id} ===")
    
    # Load weighted graph
    with open(f"molecule_{mol_id}_weighted.pkl", "rb") as f:
        G = pickle.load(f)
    
    # Compute Minimum Spanning Tree using Kruskal with alphabetical tie-breaking
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes())

    parent, rank = union_find_init(G.nodes())
    sorted_edges = sorted(
        G.edges(data=True),
        key=lambda e: (
            e[2]["weight"],
            node_sort_key(min(e[0], e[1], key=node_sort_key)),
            node_sort_key(max(e[0], e[1], key=node_sort_key)),
        ),
    )

    for u, v, data in sorted_edges:
        if union_find_union(parent, rank, u, v):
            mst.add_edge(u, v, **data)

    # Prepare molecule for visualization
    smiles_safe = smiles_to_filename(smiles_list[mol_id - 1])
    mol = Chem.MolFromSmiles(smiles_list[mol_id - 1])
    rdDepictor.Compute2DCoords(mol)
    annotate_atom_labels(mol)
    
    total_weight = sum(data['weight'] for u, v, data in mst.edges(data=True))
    ordered_edges = sorted(
        mst.edges(data=True),
        key=lambda e: (node_sort_key(e[0]), node_sort_key(e[1])),
    )
    
    print("Minimum Spanning Tree edges:")
    for u, v, data in ordered_edges:
        print(f"{u} -- {v}  (weight: {data['weight']})")

    highlight_atoms = sorted({atom_index_from_label(n) for n in mst.nodes()})
    highlight_bonds = []
    for u, v in mst.edges():
        a1 = atom_index_from_label(u)
        a2 = atom_index_from_label(v)
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(
        mol,
        size=(800, 800),
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )
    overlay_lines = [
        f"Algorithm: Kruskal MST | Molecule: {mol_id}",
        f"Total weight: {total_weight} | Edges: {mst.number_of_edges()}",
    ]
    overlay_lines.extend(format_edge_lines(ordered_edges))
    add_overlay(img, overlay_lines)

    img.save(f"molecule_{mol_id}_{smiles_safe}_kruskal_mst.png")
    print(f"MST image saved: molecule_{mol_id}_{smiles_safe}_kruskal_mst.png")
    
    print(f"Total MST weight: {total_weight}")
    print(f"Number of edges in MST: {mst.number_of_edges()}")
    print("-" * 60)

print("Kruskal's MST completed for both molecules.")