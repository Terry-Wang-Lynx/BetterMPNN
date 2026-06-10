"""Unit tests for config loading and design-scope logic (pure Python)."""

import textwrap

from bettermpnn.config import Config, DesignConfig


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_minimal_config_loads_with_defaults(tmp_path):
    path = _write(tmp_path, """
        pdb: "examples/scaffold.pdb"
        chain: "A"
    """)
    cfg = Config.from_yaml(path)
    assert cfg.pdb.endswith("scaffold.pdb")
    assert cfg.chain == "A"
    assert cfg.mode == "train"
    assert cfg.environment.type == "alphafold3"


def test_scientific_notation_coerced_to_float(tmp_path):
    path = _write(tmp_path, """
        pdb: "x.pdb"
        lr: 1e-4
        beta: 1e-2
    """)
    cfg = Config.from_yaml(path)
    assert isinstance(cfg.lr, float) and abs(cfg.lr - 1e-4) < 1e-12
    assert isinstance(cfg.beta, float) and abs(cfg.beta - 1e-2) < 1e-12


def test_seed_is_parsed_as_int(tmp_path):
    path = _write(tmp_path, """
        pdb: "x.pdb"
        seed: 42
    """)
    cfg = Config.from_yaml(path)
    assert cfg.seed == 42 and isinstance(cfg.seed, int)


def test_nested_environment_and_filter(tmp_path):
    path = _write(tmp_path, """
        pdb: "x.pdb"
        mode: sample
        num_seeds: 3
        decoy_smiles: ["CCO", "c1ccccc1"]
        filter:
          iptm_min: 0.85
        environment:
          design_chain_index: 0
          reward_weights:
            iptm: 0.6
    """)
    cfg = Config.from_yaml(path)
    assert cfg.mode == "sample"
    assert cfg.num_seeds == 3
    assert cfg.decoy_smiles == ["CCO", "c1ccccc1"]
    assert cfg.filter.iptm_min == 0.85
    assert cfg.environment.design_chain_index == 0
    assert cfg.environment.reward_weights.iptm == 0.6


def test_filter_legacy_keys_are_remapped(tmp_path):
    path = _write(tmp_path, """
        pdb: "x.pdb"
        filter:
          conf_rmsd_min: 1.5
          decoy_conf_rmsd_max: 2.0
    """)
    cfg = Config.from_yaml(path)
    # Legacy keys map onto the current field names.
    assert cfg.filter.open_rmsd_min == 1.5
    assert cfg.filter.decoy_closed_rmsd_min == 2.0


def test_effective_target_smiles_prefers_ligand(tmp_path):
    path = _write(tmp_path, """
        pdb: "x.pdb"
        target_smiles: "CCO"
        ligand:
          smiles: "c1ccccc1"
    """)
    cfg = Config.from_yaml(path)
    assert cfg.effective_target_smiles == "c1ccccc1"


def test_design_redesign_and_fixed_positions():
    d = DesignConfig(redesign_residues=["A10", "A11", "B5"])
    assert d.get_redesign_positions("A") == [10, 11]
    # Fixing is the inverse of redesign within the chain length.
    assert d.get_fixed_positions("A", 5) == [1, 2, 3, 4, 5]
    assert d.get_fixed_positions("A", 12) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]


def test_empty_design_means_design_everything():
    d = DesignConfig(redesign_residues=[])
    assert d.get_redesign_positions("A") == []
    assert d.get_fixed_positions("A", 10) == []
