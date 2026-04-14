from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from PIL import ImageDraw
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import re

print("Starting DFS from C7 for both molecules...\n")

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


def color_for_step(step: int, total: int):
    if total <= 1:
        return (0.1, 0.65, 0.2)
    ratio = step / (total - 1)
    # Green -> orange gradient across traversal order.
    return (0.15 + 0.75 * ratio, 0.65 - 0.35 * ratio, 0.2)


def annotate_dfs_order_labels(mol, visited):
    """Write atom labels with DFS order, e.g. C7(1), C5(2)."""
    step_map = {atom_index_from_label(label): step + 1 for step, label in enumerate(visited)}
    for atom in mol.GetAtoms():
        atom_label = f"{atom.GetSymbol()}{atom.GetIdx()}"
        step = step_map.get(atom.GetIdx())
        if step is not None:
            atom.SetProp("atomNote", f"{atom_label}({step})")
        else:
            atom.SetProp("atomNote", atom_label)


def format_order_lines(prefix: str, labels, chunk_size: int = 7):
    chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
    lines = []
    for i, chunk in enumerate(chunks):
        text = " -> ".join(chunk)
        if i == 0:
            lines.append(f"{prefix} {text}")
        else:
            lines.append(f"       {text}")
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


def hierarchy_pos(G, root, width=1.0, vert_gap=0.15, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    neighbors = list(G.neighbors(root))
    if parent is not None and parent in neighbors:
        neighbors.remove(parent)
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos


def draw_tree_graph(tree, visited, start, filepath, title):
    pos = hierarchy_pos(tree, start)
    labels = {node: f"{node}({i + 1})" for i, node in enumerate(visited)}

    plt.figure(figsize=(10, 6))
    nx.draw(tree, pos=pos, with_labels=False, arrows=False, node_size=900, node_color="#ff9999", edge_color="black")
    nx.draw_networkx_labels(tree, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def smiles_to_filename(smiles: str) -> str:
    for ch in r'\/:*?"<>|':
        smiles = smiles.replace(ch, "_")
    return smiles


for mol_id in [1, 2]:
    print(f"=== Molecule {mol_id} ===")
    
    # Load unweighted graph for BFS/DFS
    with open(f"molecule_{mol_id}_unweighted.pkl", "rb") as f:
        G = pickle.load(f)

    smiles_safe = smiles_to_filename(smiles_list[mol_id - 1])
    mol = Chem.MolFromSmiles(smiles_list[mol_id - 1])
    rdDepictor.Compute2DCoords(mol)

    start = "C7"

    if start not in G:
        print(f"Skipped: start node {start} does not exist in this graph")
        img = Draw.MolToImage(mol, size=(800, 800))
        add_overlay(
            img,
            [
                f"Algorithm: DFS | Molecule: {mol_id}",
                f"Start node {start} not found in graph",
            ],
        )
        img.save(f"molecule_{mol_id}_{smiles_safe}_dfs_structure.png")
        print(f"Structure image saved (no traversal): molecule_{mol_id}_{smiles_safe}_dfs_structure.png")
        print("-" * 60)
        continue
    
    # DFS implementation using stack — mark nodes visited when POPPED
    # so traversal order matches true recursive DFS.
    visited = []
    seen = set()
    # Each stack entry is (parent_label, node_label)
    stack = [(None, start)]
    traversal_edges = []

    while stack:
        parent, current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        visited.append(current)
        if parent is not None:
            traversal_edges.append((parent, current))

        neighbors = sorted(G.neighbors(current), key=node_sort_key)
        # Push in reverse so smallest-index neighbor is popped first.
        for neigh in reversed(neighbors):
            if neigh not in seen:
                stack.append((current, neigh))
    
    # Output result
    print("DFS visiting order (starting from C7):")
    print(visited)
    print(f"Total nodes visited: {len(visited)}")

    annotate_dfs_order_labels(mol, visited)

    highlight_atoms = [atom_index_from_label(node) for node in visited]
    atom_colors = {
        atom_idx: color_for_step(step, len(highlight_atoms))
        for step, atom_idx in enumerate(highlight_atoms)
    }

    highlight_bonds = []
    bond_colors = {}
    for step, (u, v) in enumerate(traversal_edges):
        a1 = atom_index_from_label(u)
        a2 = atom_index_from_label(v)
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            bond_idx = bond.GetIdx()
            highlight_bonds.append(bond_idx)
            bond_colors[bond_idx] = color_for_step(step, len(traversal_edges))

    img = Draw.MolToImage(
        mol,
        size=(800, 800),
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )

    overlay_lines = [
        f"Algorithm: DFS | Molecule: {mol_id}",
        f"Start: {start} | Visited: {len(visited)} nodes",
    ]
    overlay_lines.extend(format_order_lines("Order:", visited))
    add_overlay(img, overlay_lines)

    img.save(f"molecule_{mol_id}_{smiles_safe}_dfs_structure.png")

    tree = nx.DiGraph(traversal_edges)
    draw_tree_graph(
        tree,
        visited,
        start,
        f"molecule_{mol_id}_{smiles_safe}_dfs_tree.png",
        f"DF-tree Molecule {mol_id} (start {start})",
    )

    print(f"Structure image saved: molecule_{mol_id}_{smiles_safe}_dfs_structure.png")
    print(f"Tree image saved: molecule_{mol_id}_{smiles_safe}_dfs_tree.png")
    print("-" * 60)

print("DFS completed for both molecules.")