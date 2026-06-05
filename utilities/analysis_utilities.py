import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from typing import Dict, List, Any, Set

from utilities.data.structure import Structure
from utilities.constants import xxx_to_index, atom_types

def analyze_folder(folder_path: str, output_file: str = "structure_data_summary.json") -> Dict[str, Any]:
    """
    Original analyze_folder implementation (flat dictionary).
    """
    directory = Path(folder_path)
    npz_files = list(directory.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {folder_path}")
        return {}

    summary_dictionary = {}

    for file_path in tqdm(npz_files, desc="Analyzing Structures"):
        file_name = file_path.stem
        try:
            structure = Structure(npz_path=str(file_path))
            
            number_residues = structure.number_residues
            number_chains = structure.number_chains

            total_atoms = len(structure.atoms)
            missing_atom_count = 0
            unique_present_atoms = set()
            
            for atom in structure.atoms:
                if atom.is_present:
                    unique_present_atoms.add(atom.name)
                else:
                    missing_atom_count += 1
                    
            missing_atom_percentage = round(100 * missing_atom_count / total_atoms, 2) if total_atoms > 0 else 0
            
            has_missing_atom = missing_atom_count > 0
            has_chirality = any(atom.chirality != 0 for atom in structure.atoms)
            has_charge = any(atom.charge != 0 for atom in structure.atoms)
            
            # Residue-level checks
            missing_residue_count = sum(1 for res in structure.residues if not res.is_present)
            missing_residue_percentage = round(100 * missing_residue_count / number_residues, 2) if number_residues > 0 else 0

            non_standard_residues = sorted(list(set(res.name for res in structure.residues if not res.is_standard)))
            has_non_standard_residue = len(non_standard_residues) > 0
            has_non_present_residue = missing_residue_count > 0

            # Structural Gaps
            structural_gaps = 0
            for chain in structure.chains:
                start = chain.residue_start_index
                end = start + chain.residue_count
                chain_residues = structure.residues[start:end]
                for i in range(1, len(chain_residues)):
                    if chain_residues[i].residue_index - chain_residues[i-1].residue_index > 1:
                        structural_gaps += 1

            # Residue distribution
            if number_residues > 0:
                residue_counts = Counter(res.name for res in structure.residues)
                residue_distribution = {
                    res_name: round(100 * count / number_residues, 2) 
                    for res_name, count in residue_counts.items()
                }
            else:
                residue_distribution = {}

            # Store in summary dictionary (Original Flat Format)
            summary_dictionary[file_name] = {
                "number_residues": number_residues,
                "number_chains": number_chains,
                "has_missing_atom": has_missing_atom,
                "missing_atom_percentage": missing_atom_percentage,
                "unique_present_atoms": sorted(list(unique_present_atoms)),
                "has_chirality": has_chirality,
                "has_charge": has_charge,
                "has_non_standard_residue": has_non_standard_residue,
                "non_standard_residue_names": non_standard_residues, # Added suggestion
                "has_non_present_residue": has_non_present_residue,
                "missing_residue_percentage": missing_residue_percentage,
                "structural_gaps": structural_gaps,
                "residue_distribution": residue_distribution
            }

        except Exception as e:
            print(f"\nError processing {file_name}: {e}")

    with open(output_file, 'w') as f:
        json.dump(summary_dictionary, f, indent=4)

    return summary_dictionary

def analyze_summary_json(json_path: str, threshold: float = 20.0):
    """
    Reads the summary JSON and computes all requested statistics without reading NPZ files.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_observed_residues = set()
    all_observed_atoms = set()
    high_dist_count = 0
    total_structures = len(data)
    
    all_non_standard_found = set()

    for struct_id, info in data.items():
        # Collect residues from distribution keys
        dist = info.get("residue_distribution", {})
        all_observed_residues.update(dist.keys())
        
        # Check high distribution threshold
        if any(pct > threshold for pct in dist.values()):
            high_dist_count += 1
            
        # Collect atoms
        atoms = info.get("unique_present_atoms", [])
        all_observed_atoms.update(atoms)
        
        # Track non-standard residue names if available
        non_std = info.get("non_standard_residue_names", [])
        all_non_standard_found.update(non_std)

    # Vocabulary checks
    supported_residues = set(xxx_to_index.keys())
    supported_atoms = set(atom_types)
    
    missing_residues = sorted(list(all_observed_residues - supported_residues))
    missing_atoms = sorted(list(all_observed_atoms - supported_atoms))
    
    high_dist_pct = round(100 * high_dist_count / total_structures, 2) if total_structures > 0 else 0

    # Print Report
    print("\n" + "="*50)
    print(f"JSON SUMMARY ANALYSIS: {json_path}")
    print("="*50)
    print(f"Total Structures in JSON: {total_structures}")
    
    print("\n--- Vocabulary Coverage (Missing from constants.py) ---")
    print(f"Missing Residues: {missing_residues if missing_residues else 'None'}")
    print(f"Missing Atoms:    {missing_atoms if missing_atoms else 'None'}")
    
    print("\n--- Distribution Statistics ---")
    print(f"Structures with any residue >{threshold}%: {high_dist_pct}%")
    
    if all_non_standard_found:
        print("\n--- Non-Standard Residues Identified ---")
        print(f"Names: {sorted(list(all_non_standard_found))}")
    
    print("="*50 + "\n")

    return {
        "missing_residues": missing_residues,
        "missing_atoms": missing_atoms,
        "high_dist_pct": high_dist_pct
    }
