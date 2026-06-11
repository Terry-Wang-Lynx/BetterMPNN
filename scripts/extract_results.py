"""Flatten a sampling 'passed/' tree into one directory of renamed CIFs.

Each passing candidate lives in 'stepN_varM_seedK/' with target / apo / decoy
structures; this collects them as '<step>_<var>_<seed>-<role>.cif'.
"""
import os
import shutil
import re
import argparse


def extract_cifs(source_base, dest_base):
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)

    dir_pattern = re.compile(r"step(\d+)_var(\d+)_seed(\d+)")
    decoy_pattern = re.compile(r"bound_decoy_(\d+)")

    count = 0
    for subdir in os.listdir(source_base):
        match = dir_pattern.search(subdir)
        if not match:
            continue
        step, var, seed = match.groups()
        prefix = f"{step}_{var}_{seed}"

        subdir_path = os.path.join(source_base, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for f in os.listdir(subdir_path):
            src_file = os.path.join(subdir_path, f)
            new_name = None
            if "bound_target.cif" in f:
                new_name = f"{prefix}-target.cif"
            elif "predicted_apo.cif" in f:
                new_name = f"{prefix}-open.cif"
            else:
                d = decoy_pattern.search(f)
                if d and f.endswith(".cif"):
                    new_name = f"{prefix}-decoy{int(d.group(1)) + 1}.cif"

            if new_name:
                shutil.copy2(src_file, os.path.join(dest_base, new_name))
                count += 1

    print(f"Extraction complete.")
    print(f"Source: {source_base}")
    print(f"Destination: {dest_base}")
    print(f"Total files copied: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Rename BetterMPNN CIF results")
    parser.add_argument("--input", required=True, help="Path to 'passed' directory")
    parser.add_argument("--output", required=True, help="Destination directory for flat CIFs")
    
    args = parser.parse_args()
    extract_cifs(args.input, args.output)
