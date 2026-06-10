import os
import shutil
import re
import argparse

def extract_cifs(source_base, dest_base):
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)

    # Pattern for directory: step19_var2_seed48
    dir_pattern = re.compile(r"step(\d+)_var(\d+)_seed(\d+)")
    
    count = 0
    # Walk through each subdirectory in 'passed'
    for subdir in os.listdir(source_base):
        match = dir_pattern.search(subdir)
        if match:
            step, var, seed = match.groups()
            prefix = f"{step}_{var}_{seed}"
            
            subdir_path = os.path.join(source_base, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            files = os.listdir(subdir_path)
            
            for f in files:
                # We handle .cif (predicted) and potentially reference files
                src_file = os.path.join(subdir_path, f)
                new_name = None
                
                # Naming Logic
                if "bound_target.cif" in f:
                    new_name = f"{prefix}-target.cif"
                elif "predicted_apo.cif" in f:
                    new_name = f"{prefix}-open.cif"
                elif "bound_decoy_0" in f:
                    new_name = f"{prefix}-decoy1.cif"
                elif "bound_decoy_1" in f:
                    new_name = f"{prefix}-decoy2.cif"
                elif "bound_decoy_2" in f:
                    new_name = f"{prefix}-decoy3.cif"
                
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
