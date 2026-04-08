"""Structure format conversion utilities."""

import logging

logger = logging.getLogger(__name__)


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """Convert a CIF file to PDB format using biotite."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io

    f = pdbx.PDBxFile.read(cif_path)
    structure = pdbx.get_structure(f, model=1)

    f_pdb = pdb_io.PDBFile()
    f_pdb.set_structure(structure)
    f_pdb.write(pdb_path)
    logger.info(f"Converted {cif_path} -> {pdb_path}")
