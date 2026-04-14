from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

smiles = "CCN(CC)C(=O)c1cccnc1"
mol = Chem.MolFromSmiles(smiles)
Chem.rdDepictor.Compute2DCoords(mol)  # get 2D coordinates for drawing

# Create a drawer (SVG output here; you can also use PNG)
drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
opts = drawer.drawOptions()

# Add labels for each atom
for atom in mol.GetAtoms():
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    opts.atomLabels[idx] = f"{symbol}{idx}"   # e.g. C0, C1, O6

drawer.DrawMolecule(mol)
drawer.FinishDrawing()

svg = drawer.GetDrawingText()
with open("./molecule_labeled.svg", "w") as f:
    f.write(svg)