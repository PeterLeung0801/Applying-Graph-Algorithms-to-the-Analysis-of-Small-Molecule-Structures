from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from PIL import ImageDraw
import networkx as nx
import pickle
import re

print("Starting Dijkstra's algorithm (C0 to C7) for both molecules...\n")

smiles_list = [
    "CCN(CC)C(=O)c1cccnc1",      # Molecule 1
    "CCN1C(=O)NC(c2ccccc2)C1=O"  # Molecule 2
]


def atom_index_from_label(label: str) -> int:
    match = re.search(r"(\d+)$", label)
    if not match:
        raise ValueError(f"Cannot parse atom index from label: {label}")
    return int(match.group(1))


def annotate_path_order_labels(mol, path):
    """Write path atom labels with order, e.g. C0(1), C1(2)."""
    step_map = {atom_index_from_label(label): step + 1 for step, label in enumerate(path)}
    for atom in mol.GetAtoms():
        atom_label = f"{atom.GetSymbol()}{atom.GetIdx()}"
        step = step_map.get(atom.GetIdx())
        if step is not None:
            atom.SetProp("atomNote", f"{atom_label}({step})")
        else:
            atom.SetProp("atomNote", atom_label)


def annotate_atom_labels(mol):
    """Write the atom label (e.g. C7, N2) next to each atom."""
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", f"{atom.GetSymbol()}{atom.GetIdx()}")


def node_sort_key(label: str):
    match = re.match(r"([A-Za-z]+)(\d+)$", label)
    if not match:
        return (label, -1)
    return (match.group(1), int(match.group(2)))


def dijkstra_shortest_path(G, source, target):
    import heapq

    distances = {node: float('inf') for node in G.nodes()}
    previous = {node: None for node in G.nodes()}
    distances[source] = 0

    queue = [(0, node_sort_key(source), source)]
    visited = set()

    while queue:
        dist, _, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == target:
            break

        for neighbor in sorted(G.neighbors(current), key=node_sort_key):
            weight = G[current][neighbor].get('weight', 1)
            new_dist = dist + weight
            if new_dist < distances[neighbor] or (new_dist == distances[neighbor] and node_sort_key(current) < node_sort_key(previous[neighbor] or neighbor)):
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(queue, (new_dist, node_sort_key(neighbor), neighbor))

    if distances[target] == float('inf'):
        raise nx.NetworkXNoPath(f"No path between {source} and {target}")

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()
    return path, distances[target]


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


def smiles_to_filename(smiles: str) -> str:
    for ch in r'\/:*?"<>|':
        smiles = smiles.replace(ch, "_")
    return smiles


for mol_id in [1, 2]:
    print(f"=== Molecule {mol_id} ===")
    
    # Load weighted graph
    with open(f"molecule_{mol_id}_weighted.pkl", "rb") as f:
        G = pickle.load(f)
    
    # Prepare molecule image canvas
    smiles_safe = smiles_to_filename(smiles_list[mol_id - 1])
    mol = Chem.MolFromSmiles(smiles_list[mol_id - 1])
    rdDepictor.Compute2DCoords(mol)
    annotate_atom_labels(mol)

    source = "C0"
    target = "C7"

    if source not in G or target not in G:
        print(f"Skipped: source ({source}) or target ({target}) does not exist in this graph")
        img = Draw.MolToImage(mol, size=(800, 800))
        add_overlay(
            img,
            [
                f"Algorithm: Dijkstra | Molecule: {mol_id}",
                f"Missing source/target: {source} -> {target}",
            ],
        )
        img.save(f"molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
        print(f"Image saved (no highlighted path): molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
        print("-" * 60)
        continue

    # Dijkstra shortest path from source to target with alphabetical tie-breaking
    try:
        path, distance = dijkstra_shortest_path(G, source, target)
        
        print("Shortest path from C0 to C7:")
        print(" → ".join(path))
        print(f"Total distance (cost): {distance}")

        annotate_path_order_labels(mol, path)

        highlight_atoms = [atom_index_from_label(node) for node in path]
        highlight_bonds = []
        for i in range(len(path) - 1):
            a1 = atom_index_from_label(path[i])
            a2 = atom_index_from_label(path[i + 1])
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
            f"Algorithm: Dijkstra | Molecule: {mol_id}",
            f"Path cost: {distance}",
        ]
        overlay_lines.extend(format_order_lines("Path:", path))
        add_overlay(img, overlay_lines)
        img.save(f"molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
        print(f"Path image saved: molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
        
    except nx.NetworkXNoPath:
        print("No path found between C0 and C7")
        img = Draw.MolToImage(mol, size=(800, 800))
        add_overlay(
            img,
            [
                f"Algorithm: Dijkstra | Molecule: {mol_id}",
                f"No path found: {source} -> {target}",
            ],
        )
        img.save(f"molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
        print(f"Image saved (no highlighted path): molecule_{mol_id}_{smiles_safe}_dijkstra_path.png")
    
    print("-" * 60)

print("Dijkstra completed for both molecules.")