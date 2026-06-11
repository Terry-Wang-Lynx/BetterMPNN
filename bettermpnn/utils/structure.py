"""Structure format conversion and coordinate extraction utilities."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """Convert a CIF file to PDB format using biotite."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io

    if hasattr(pdbx, "CIFFile"):  # biotite >= 1.0
        structure = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1)
    else:  # legacy biotite
        structure = pdbx.get_structure(pdbx.PDBxFile.read(cif_path), model=1)

    f_pdb = pdb_io.PDBFile()
    f_pdb.set_structure(structure)
    f_pdb.write(pdb_path)
    logger.info(f"Converted {cif_path} -> {pdb_path}")


def extract_ca_from_pdb(pdb_path: str, chain_id: str) -> np.ndarray:
    """Extract Cα coordinates from a PDB file for a specific chain.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier (e.g., "A").

    Returns:
        Cα coordinates as numpy array of shape (N, 3).
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain = line[21].strip()
            if atom_name == "CA" and chain == chain_id:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    ca = np.array(coords, dtype=np.float64) if coords else np.empty((0, 3))  # (N, 3)
    logger.debug(f"Extracted {len(ca)} Cα atoms from {pdb_path} chain {chain_id}")
    return ca


def extract_ca_from_cif(cif_path: str, chain_id: str) -> np.ndarray:
    """Extract Cα coordinates from a CIF file for a specific chain.

    Parses mmCIF _atom_site loop directly without heavy dependencies.

    Args:
        cif_path: Path to CIF file.
        chain_id: Chain identifier (e.g., "A").

    Returns:
        Cα coordinates as numpy array of shape (N, 3).
    """
    coords = []
    in_atom_site = False
    column_names = []
    label_atom_id_idx = -1
    label_asym_id_idx = -1
    auth_asym_id_idx = -1
    cartn_x_idx = -1
    cartn_y_idx = -1
    cartn_z_idx = -1
    group_pdb_idx = -1

    with open(cif_path) as f:
        for line in f:
            line = line.rstrip()

            # Detect start of _atom_site loop
            if line.startswith("loop_"):
                in_atom_site = False
                column_names = []
                continue

            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].strip()
                column_names.append(col_name)
                continue

            if in_atom_site and not line.startswith("_") and line.strip():
                # First data line: resolve column indices
                if label_atom_id_idx < 0:
                    for i, name in enumerate(column_names):
                        if name == "label_atom_id":
                            label_atom_id_idx = i
                        elif name == "label_asym_id":
                            label_asym_id_idx = i
                        elif name == "auth_asym_id":
                            auth_asym_id_idx = i
                        elif name == "Cartn_x":
                            cartn_x_idx = i
                        elif name == "Cartn_y":
                            cartn_y_idx = i
                        elif name == "Cartn_z":
                            cartn_z_idx = i
                        elif name == "group_PDB":
                            group_pdb_idx = i

                # Parse data line
                if line.startswith("#") or line.startswith("loop_"):
                    in_atom_site = False
                    continue

                parts = line.split()
                if len(parts) <= max(label_atom_id_idx, cartn_z_idx):
                    continue

                # Filter: only ATOM records, CA atoms, matching chain
                if group_pdb_idx >= 0 and parts[group_pdb_idx] != "ATOM":
                    continue

                atom_name = parts[label_atom_id_idx]
                # Try auth_asym_id first, fall back to label_asym_id
                chain_col = auth_asym_id_idx if auth_asym_id_idx >= 0 else label_asym_id_idx
                chain = parts[chain_col] if chain_col >= 0 else ""

                if atom_name == "CA" and chain == chain_id:
                    x = float(parts[cartn_x_idx])
                    y = float(parts[cartn_y_idx])
                    z = float(parts[cartn_z_idx])
                    coords.append([x, y, z])

            elif in_atom_site and (line.startswith("#") or line == ""):
                in_atom_site = False

    ca = np.array(coords, dtype=np.float64) if coords else np.empty((0, 3))
    logger.debug(f"Extracted {len(ca)} Cα atoms from {cif_path} chain {chain_id}")
    return ca
