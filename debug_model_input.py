import torch
from utilities.data.input import ModelInput

def main():
    print("Testing ModelInput...")
    
    # Paths for a specific sample from OpenFold data
    base_path = "data_examples/openfold"
    protein_id = "P97973"
    
    structure_path = f"{base_path}/structures/{protein_id}.npz"
    msa_path = f"{base_path}/raw_msa/{protein_id}.a3m"
    record_path = f"{base_path}/records/{protein_id}.json"
    
    print(f"Using Structure: {structure_path}")
    print(f"Using MSA: {msa_path}")
    
    # Initialize the model input with typical hyper-parameters
    try:
        model_input = ModelInput(
            structure_path=structure_path,
            msa_path=msa_path,
            record_path=record_path,
            acceptance_slope_start=256,
            acceptance_slope_end=512,
            residue_crop_size=256,
            emphasize_beginning_crops=False,
            distribution_threshold=90,
            maximum_cluster_sequences=128,
            maximum_extra_msa_sequences=1024,
            mask_probability=0.15,
            debug=True,
            device=torch.device("cpu"),
            dtype=torch.float32
        )
        
        print("\nModelInput initialized successfully!")
        
        # Test feature extraction
        print("\nExtracting data...")
        data = model_input.get_data(number_samples=3)
        
        print("\nExtracted Data Shapes:")
        for key, tensor in data.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{key}: {tensor.shape}")
            else:
                print(f"{key}: {type(tensor)}")
                
    except Exception as e:
        print(f"\nError occurred during initialization or feature extraction:\n{e}")

if __name__ == "__main__":
    main()
