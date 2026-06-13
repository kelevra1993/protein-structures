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

- **uv**:

We have transitioned to using `uv` as our package manager instead of `Poetry`. `uv` is written in Rust and provides phenomenally faster dependency resolution. **Crucially, `uv` acts as a full toolchain manager; you do not need to install Python manually.**

It will automatically fetch the required Python 3.12.10 version in the background.

- **MMseqs2**:

We use MMseqs2 for rapid protein sequence searching and clustering. Generating deep Multiple Sequence Alignments (MSAs) is critical for AlphaFold's accuracy, and MMseqs2 allows us to search massive sequence databases orders of magnitude faster than traditional tools like HHblits.

### Setup Instructions

1. Clone the repository and navigate into it:
   ```bash
   https://github.com/kelevra1993/protein-structures.git
   cd protein-structures
   ```

2. Create the virtual environment (`.venv`) and install all dependencies using `uv`:
   ```bash
   uv sync
   ```
   *(Note: Because of our `pyproject.toml` configuration, running `uv sync` will automatically download Python 3.12.10 if you do not already have it, create the `.venv` folder, and install the exact package versions specified in the lockfile).*

