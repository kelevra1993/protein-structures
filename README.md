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

