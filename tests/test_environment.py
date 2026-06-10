"""Tests for AlphaFold3 environment parsing/filtering and example consistency."""

import json

from bettermpnn.config import EnvironmentConfig, FilterConfig
from bettermpnn.environment.alphafold3 import AlphaFold3Environment, PAE_MAX


def _make_env(tmp_path, design_chain_index=0):
    import pathlib
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    template = {
        "name": "t",
        "modelSeeds": [1],
        "sequences": [
            {"protein": {"id": ["A"], "sequence": "AAAA",
                         "unpairedMsa": ">A\nAAAA", "pairedMsa": ">A\nAAAA"}},
            {"protein": {"id": ["B"], "sequence": "CCCC"}},
        ],
    }
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(template))
    cfg = EnvironmentConfig(template_json=str(tpl), design_chain_index=design_chain_index)
    return AlphaFold3Environment(cfg, output_dir=str(tmp_path / "out"))


def test_check_filter_hard_thresholds():
    fc = FilterConfig(iptm_min=0.8, ptm_min=0.0, pae_max=31.75)
    passing = {"iptm": 0.9, "ptm": 0.5, "mean_pae": 5.0}
    failing = {"iptm": 0.5, "ptm": 0.5, "mean_pae": 5.0}
    assert AlphaFold3Environment._check_filter(passing, None, None, None, fc) is True
    assert AlphaFold3Environment._check_filter(failing, None, None, None, fc) is False


def test_check_filter_conformational_and_fold():
    fc = FilterConfig(iptm_min=0.0, open_rmsd_min=2.0, closed_rmsd_max=2.5, local_drmsd_max=3.0)
    m = {"iptm": 0.95, "ptm": 0.9, "mean_pae": 5.0}
    # open_rmsd below the minimum required conformational change -> reject
    assert AlphaFold3Environment._check_filter(m, 1.0, 1.0, 1.0, fc) is False
    # closed_rmsd above max -> reject
    assert AlphaFold3Environment._check_filter(m, 3.0, 5.0, 1.0, fc) is False
    # local_drmsd above max (fold broken) -> reject
    assert AlphaFold3Environment._check_filter(m, 3.0, 1.0, 9.0, fc) is False
    # all within bounds -> pass
    assert AlphaFold3Environment._check_filter(m, 3.0, 1.0, 1.0, fc) is True


def test_check_filter_none_config_passes():
    assert AlphaFold3Environment._check_filter({"iptm": 0.0}, None, None, None, None) is True


def test_extract_mean_pae_is_chain_aware(tmp_path):
    data = {"chain_pair_pae_min": [[0, 10, 20], [12, 0, 30], [22, 32, 0]]}
    env0 = _make_env(tmp_path / "a", design_chain_index=0)
    env1 = _make_env(tmp_path / "b", design_chain_index=1)
    # design chain 0 vs {1,2}: mean(10,12,20,22) = 16
    assert abs(env0._extract_mean_pae(data) - 16.0) < 1e-9
    # design chain 1 vs {0,2}: mean(12,10,30,32) = 21
    assert abs(env1._extract_mean_pae(data) - 21.0) < 1e-9


def test_extract_mean_pae_fallback(tmp_path):
    env = _make_env(tmp_path, design_chain_index=0)
    assert env._extract_mean_pae({}) == PAE_MAX


def test_calculate_reward_clamped_and_weighted(tmp_path):
    env = _make_env(tmp_path, design_chain_index=0)
    # has_clash applies the clash penalty; reward stays within [0, 1]
    clean = env._calculate_reward({"iptm": 0.9, "ptm": 0.9, "mean_pae": 0.0, "has_clash": 0.0})
    clashed = env._calculate_reward({"iptm": 0.9, "ptm": 0.9, "mean_pae": 0.0, "has_clash": 1.0})
    assert 0.0 <= clashed <= clean <= 1.0


def test_design_chain_index_out_of_range_raises(tmp_path):
    env = _make_env(tmp_path, design_chain_index=5)
    try:
        env._create_input_json("AAAA", "job")
        assert False, "expected ValueError for out-of-range design_chain_index"
    except ValueError:
        pass


def test_example_scaffold_matches_template():
    """The bundled GZMK example must stay self-consistent: the redesigned
    chain length in the scaffold PDB must equal the template sequence length."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    pdb = root / "examples" / "scaffold.pdb"
    tpl = root / "configs" / "af3_template.json"
    if not (pdb.exists() and tpl.exists()):
        return  # example assets not present; skip
    ca_a = 0
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
            ca_a += 1
    template = json.loads(tpl.read_text())
    seq0 = template["sequences"][0]["protein"]["sequence"]
    assert ca_a == len(seq0), f"scaffold chain A has {ca_a} residues but template[0] has {len(seq0)}"
