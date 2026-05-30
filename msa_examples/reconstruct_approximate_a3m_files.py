import os
import glob
from pathlib import Path
import numpy as np


def reconstruct_boltz_a3m(npz_path, output_a3m_path, filler_char="x"):
    """Reconstructs a valid .a3m MSA file from Boltz .npz records

    using the model's exact native token constants.
    """
    # 1. Rebuild Boltz's precise internal tokens array
    canonical_tokens = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
    ]

    # Stacking list items exactly like const.py: tokens = ["<pad>", "-", *canonical_tokens, ...]
    tokens = ["<pad>", "-"] + canonical_tokens

    # Dictionary translation mapping for 3-letter tokens to 1-letter codes
    three_to_one = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "UNK": "X",
        "-": "-",
        "<pad>": "-",  # Treat tensor pads as gaps if they appear in sequences
    }

    # 2. Extract and parse arrays from the file
    with np.load(npz_path, allow_pickle=True) as data:
        sequences = data["sequences"]
        residues = data["residues"]
        deletions = data["deletions"]

        with open(output_a3m_path, "w") as f:
            for seq_meta in sequences:
                seq_idx = seq_meta["seq_idx"]
                taxonomy = seq_meta["taxonomy"]
                res_start = seq_meta["res_start"]
                res_end = seq_meta["res_end"]
                del_start = seq_meta["del_start"]
                del_end = seq_meta["del_end"]

                # Extract tokens for this sequence slice
                seq_tokens = residues[res_start:res_end]

                # 3. Translate using the explicit layout from const.py
                seq_chars = []
                for t in seq_tokens:
                    token_id = int(t["res_type"])

                    # Protect against array edge bounds
                    if token_id < len(tokens):
                        three_letter = tokens[token_id]
                        char = three_to_one.get(three_letter, "X")
                    else:
                        char = "X"  # Fallback for unexpected trailing structural identifiers

                    seq_chars.append(char)

                # 4. Handle insertions relative to the query structure
                seq_dels = deletions[del_start:del_end]
                sorted_dels = sorted(
                    seq_dels, key=lambda x: x["res_idx"], reverse=True
                )

                for del_record in sorted_dels:
                    r_idx = int(del_record["res_idx"])
                    del_count = int(del_record["deletion"])

                    insertion_str = filler_char.lower() * del_count
                    insert_position = r_idx

                    if 0 <= insert_position <= len(seq_chars):
                        seq_chars.insert(insert_position, insertion_str)

                # 5. Compile and output FASTA record format block
                final_sequence = "".join(seq_chars)
                f.write(f">seq_{seq_idx}_tax_{taxonomy}\n")
                f.write(f"{final_sequence}\n")

    print(f"File successfully created: {output_a3m_path}")


def convert_extension(file_path: str, before: str, after: str) -> str:
    return file_path.replace(f".{before}", f".{after}")


# --- Execution ---
if __name__ == "__main__":

    msa_examples_folder = Path(os.getcwd())

    open_fold_folder = msa_examples_folder / "open_fold_msa"

    for file_path in glob.glob(str(open_fold_folder / "*.npz")):
        input_file = file_path
        output_file = convert_extension(file_path=file_path, before="npz", after="a3m")
        reconstruct_boltz_a3m(input_file, output_file, filler_char="x")

    rcsb_folder = msa_examples_folder / "rcsb_msa"

    for file_path in glob.glob(str(rcsb_folder / "*.npz")):
        input_file = file_path
        output_file = convert_extension(file_path=file_path, before="npz", after="a3m")
        reconstruct_boltz_a3m(input_file, output_file, filler_char="x")
