# Protein Structures

## Table of Contents
1. [Description of the Project](#description-of-the-project)
2. [Prerequisites](#prerequisites)
3. [Data](#data)
   - [Data Retrieval](#data-retrieval)
   - [Data Splitting](#data-splitting)
4. [Project Configurations](#project-configurations)
5. [Project Structure](#project-structure)
6. [Running First Training experiment](#running-first-training-experiment)
7. [Visualisation](#visualisation)
8. [Tests](#tests)

---

## Description of the Project
The primary goal of this project is to explore, implement, and train the AlphaFold II architecture utilizing an NVIDIA DGX Spark with 128GB of unified memory.

Given the massive scale and complexity of the original AlphaFold model, a core focus has been placed on maintaining rigorous test coverage across all individual modules.

This robust testing infrastructure provides the foundation needed to safely experiment with novel structural designs and alternative implementations in order to make the model fit into the device it is being trained on.

## Prerequisites

Before getting started, ensure you have the following tools installed on your system:

- **uv** [Package Management]:

We have transitioned to using `uv` as our package manager instead of `Poetry`. `uv` is written in Rust and provides phenomenally faster dependency resolution. **Crucially, `uv` acts as a full toolchain manager; you do not need to install Python manually.**

It will automatically fetch the required Python 3.12.10 version in the background.

- **MMseqs2** [Data Clustering]:

We use MMseqs2 to cluster our protein sequence data based on sequence identity. This clustering step is crucial before splitting our data into training, validation, and testing sets, as it prevents data leakage by ensuring closely related sequences are grouped together and not scattered across different splits.

### Setup Instructions

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/kelevra1993/protein-structures.git
   cd protein-structures
   ```

2. Create the virtual environment (`.venv`) and install all dependencies using `uv`:
   ```bash
   uv sync
   ```
   *(Note: Because of our `pyproject.toml` configuration, running `uv sync` will automatically download Python 3.12.10 if you do not already have it, create the `.venv` folder, and install the exact package versions specified in the lockfile).*

## Data

### Data Retrieval

We use training data that has been graciously provided and pre-processed by the team behind [Boltz](https://github.com/jwohlwend/boltz).

For full details on their data processing pipeline, you can refer to their [training documentation](https://github.com/jwohlwend/boltz/blob/main/docs/training.md).

**Storage Warning**: Please ensure you have at least **500 GB of free storage space** available. While the final extracted data will take up less room, you need significant overhead to accommodate both downloading the large compressed files and decompressing them simultaneously before the archives can be deleted.

To download the pre-processed OpenFold structures, run the following commands:

```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_processed_targets.tar
tar -xf openfold_processed_targets.tar
rm openfold_processed_targets.tar
```

Next, download the raw OpenFold Multiple Sequence Alignments (MSAs).
*(Note: This specific file is 88GB compressed, and will temporarily require an additional 88GB of space to extract before you can delete the archive).*

```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_raw_msa.tar
tar -xf openfold_raw_msa.tar
rm openfold_raw_msa.tar
```

### Data Splitting

To ensure the model learns generalized representations and to prevent data leakage during training, we must split our dataset into training and validation sets carefully. Randomly splitting individual proteins is not sufficient, as highly similar sequences (homologs) could end up in both sets, causing the model to overfit and perform poorly on truly unseen data.

To solve this, we use a custom pipeline that leverages **MMseqs2** to cluster the proteins before splitting them.

#### The Splitting Pipeline

The end-to-end splitting process is handled by `utilities/data/data_splitter/run_data_splitter.py`. It performs the following steps:

1. **Sequence Extraction**: It iterates through all the `.a3m` Multiple Sequence Alignment files and extracts the query sequence (the main protein structure we want to predict).
2. **Clustering (MMseqs2)**: It groups these query sequences into clusters based on sequence identity. By default, we use a **40% sequence identity threshold** (`--min_identity 0.4`). This means that any two proteins sharing more than 40% of their sequence are grouped into the same cluster.
3. **Data Splitting**: It splits the data at the *cluster level*, rather than the individual protein level. By default, **80% of the clusters** are assigned to the training set, and the remaining **20%** are assigned to the validation set (`--train_ratio 0.8`).

#### Running the Data Splitter

You can run the data splitter using the following bash command. Make sure to provide the paths to your extracted `.a3m` files and the desired output folder:

```bash
uv run python utilities/data/data_splitter/run_data_splitter.py \
    --a3m_folder path/to/openfold/raw_msa \
    --output_folder path/to/output_dataset_splits \
    --min_identity 0.4 \
    --train_ratio 0.8 \
    --seed 42
```

**Outputs:**
Once complete, the script will generate several files in your `--output_folder`, most importantly:
- `Train.json`: Contains the mapping of clusters to sequence IDs for the training set.
- `Validation.json`: Contains the mapping of clusters to sequence IDs for the validation set.

These JSON files will be referenced in your project configuration (e.g., `cuda_configuration.yaml`) to instruct the data loaders on which proteins to use during the respective training phases.

## Project Configurations
<!-- Content will be added here -->

## Project Structure

```bash
.
├── README.md
├── architecture_modules
│   ├── attention_module
│   │   └── multi_head_attention.py
│   ├── distogram_module
│   │   └── distogram_module.py
│   ├── evoformer_module
│   │   ├── evoformer.py
│   │   ├── msa_stack.py
│   │   └── pair_stack.py
│   ├── lddt_module
│   │   └── lddt_module.py
│   └── structure_module
│       ├── invariant_point_attention_module.py
│       └── structure_module.py
├── configurations
│   ├── cuda_configuration.yaml
│   ├── mps_configuration.yaml
│   └── template_configuration.yaml
├── data_examples
│   ├── open_fold_sequences.fasta
│   └── openfold
│       ├── raw_msa
│       │   ├── P90561.a3m
│       │   ├── P90662.a3m
│       │   ├── *** ***
│       │   ├── P97976.a3m
│       │   └── P9WI00.a3m
│       ├── records
│       │   ├── P90561.json
│       │   ├── P90662.json
│       │   ├── *** ***
│       │   ├── P97976.json
│       │   └── P9WI00.json
│       └── structures
│           ├── P90561.npz
│           ├── P90662.npz
│           ├── *** ***
│           ├── P97976.npz
│           └── P9WI00.npz
├── embedders
│   ├── extra_msa_embedder.py
│   ├── input_embedder.py
│   └── recycling_embedder.py
├── feature_extraction
│   └── extractor.py
├── full_model
│   └── model.py
├── main.py
├── trainer
│   └── trainer.py
├── utilities
│   ├── constants.py
│   ├── data
│   │   ├── data_precomputer
│   │   │   ├── main.py
│   │   │   └── precompute_utilities.py
│   │   ├── data_splitter
│   │   │   ├── data_splitting_utilities.py
│   │   │   └── run_data_splitter.py
│   │   ├── dataloader.py
│   │   ├── input.py
│   │   ├── msa.py
│   │   └── structure.py
│   ├── geometry_utilities.py
│   ├── loss_utilities.py
│   ├── os_utilities.py
│   ├── tensor_utilities.py
│   └── visualization_utilities.py
├── tests
│   ├── architecture_modules
│   │   ├── attention_module
│   │   │   ├── reference_values
│   │   │   └── test_multi_head_attention.py
│   │   ├── distogram_module
│   │   │   └── test_distogram_module.py
│   │   ├── evoformer_module
│   │   │   ├── reference_values
│   │   │   ├── test_evoformer_module.py
│   │   │   ├── test_msa_stack.py
│   │   │   └── test_pair_stack.py
│   │   ├── lddt_module
│   │   │   ├── reference_values
│   │   │   └── test_lddt_module.py
│   │   └── structure_module
│   │       ├── reference_values
│   │       ├── test_invariant_point_attention_module.py
│   │       └── test_structure_module.py
│   ├── embedders
│   │   ├── extra_msa_embedder
│   │   ├── input_embedder
│   │   ├── recycling_embedder
│   │   ├── test_extra_msa_embedder.py
│   │   ├── test_input_embedder.py
│   │   └── test_recycling_embedder.py
│   ├── feature_extraction
│   │   ├── multiple_sequence_alignement.a3m
│   │   ├── reference_values
│   │   │   ├── input_extra_msa_feature.pt
│   │   │   └── input_msa_feature.pt
│   │   └── test_extractor.py
│   ├── full_model
│   │   ├── reference_values
│   │   │   ├── raw_msa
│   │   │   ├── records
│   │   │   └── structures
│   │   └── test_model.py
│   └── utilities
│       ├── data
│       │   ├── reference_values
│       │   │   ├── raw_msa
│       │   │   ├── records
│       │   │   └── structures
│       │   ├── test_dataloader.py
│       │   ├── test_input.py
│       │   ├── test_split.json
│       │   └── test_structure.py
│       ├── loss_utilities
│       │   ├── reference_values
│       │   └── test_loss_utilities.py
│       └── testing_utilities.py
├── pyproject.toml
└── uv.lock
```

## Running First Training experiment
<!-- Content will be added here -->

## Visualisation
<!-- Content will be added here -->

## Tests
<!-- Content will be added here -->
