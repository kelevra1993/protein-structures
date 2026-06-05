import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from utilities.data.structure import Structure


def analyze_folder(folder_path: str, output_file: str = "structure_data_summary.json"):
    """
    Analyzes all .npz structure files in a given directory and generates a summary JSON.
    """
    directory = Path(folder_path)
    npz_files = list(directory.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {folder_path}")
        return

    summary_dictionary = {}

    for file_path in tqdm(npz_files, desc="Analyzing Structures"):
        file_name = file_path.stem  # Gets 'P90561' from 'P90561.npz'

        try:
            structure = Structure(npz_path=str(file_path))

            # 1. Sequence Length & Chain Count
            number_residues = structure.number_residues
            number_chains = structure.number_chains

            # 2. Missing data percentages and Atom-level checks
            total_atoms = len(structure.atoms)
            missing_atom_count = sum(1 for atom in structure.atoms if not atom.is_present)
            missing_atom_percentage = round(100 * missing_atom_count / total_atoms, 2) if total_atoms > 0 else 0

            has_missing_atom = missing_atom_count > 0
            has_chirality = any(atom.chirality != 0 for atom in structure.atoms)
            has_charge = any(atom.charge != 0 for atom in structure.atoms)

            # Residue-level checks
            missing_residue_count = sum(1 for res in structure.residues if not res.is_present)
            missing_residue_percentage = round(100 * missing_residue_count / number_residues, 2) if number_residues > 0 else 0

            has_non_standard_residue = any(not res.is_standard for res in structure.residues)
            has_non_present_residue = missing_residue_count > 0

            # 3. Structural Gaps (Iterating per chain to avoid false gaps at chain boundaries)
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

            # Store in summary dictionary
            summary_dictionary[file_name] = {
                "number_residues": number_residues,
                "number_chains": number_chains,
                "has_missing_atom": has_missing_atom,
                "missing_atom_percentage": missing_atom_percentage,
                "has_chirality": has_chirality,
                "has_charge": has_charge,
                "has_non_standard_residue": has_non_standard_residue,
                "has_non_present_residue": has_non_present_residue,
                "missing_residue_percentage": missing_residue_percentage,
                "structural_gaps": structural_gaps,
                "residue_distribution": residue_distribution
            }

        except Exception as e:
            print(f"\nError processing {file_name}: {e}")

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary_dictionary, f, indent=4)

    print(f"\nAnalysis complete. Summary saved to {output_file}")


if __name__ == "__main__":
    # Example usage targeting your openfold structures
    analyze_folder("data_examples/openfold/structures")
