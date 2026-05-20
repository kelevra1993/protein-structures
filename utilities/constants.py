"""
File that contains all constants that will be used in the project
"""
# TODO ADD THREE LETTER ENCODING EASIER FOR FUTURE WORK
# TODO CONSIDER ADDING INTEGER INDICES TO AMINO ACID RESIDUES DIRECTLY ?

# Canonical Amino Acid Residues (20 amino acids)
canonical_amino_acid_residues = ["A", "R",
                                 "N", "D",
                                 "C", "Q",
                                 "E", "G",
                                 "H", "I",
                                 "L", "K",
                                 "M", "F",
                                 "P", "S",
                                 "T", "W",
                                 "Y", "V"]

# Include Unknown Amino Acid as 'X' : Used For Input Sequence Feature
# Including amino acids like selenocysteine, Pyrrolysine, ...e.t.c
all_amino_acid_residues = canonical_amino_acid_residues + ["X"]

# Including Gaps
gapped_amino_acid_residues = all_amino_acid_residues + ["-"]
