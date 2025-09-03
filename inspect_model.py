import torch
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# Path to your checkpoint file
ckpt_path = "models/new_tft_irradiance.ckpt"

try:
    # Load the checkpoint onto the CPU to avoid potential GPU issues
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print(f"Successfully loaded checkpoint from: {ckpt_path}\n")

    # The model's learned parameters and encoders are often in 'hyper_parameters'
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        
        # In pytorch-forecasting, dataset parameters are nested inside
        if 'dataset_parameters' in hparams:
            dataset_params = hparams['dataset_parameters']
            
            if 'categorical_encoders' in dataset_params:
                encoders = dataset_params['categorical_encoders']
                print("✅ Found categorical encoders in the model file.")
                
                if 'group_id' in encoders:
                    group_id_encoder = encoders['group_id']
                    
                    # The learned categories are stored in the .classes_ attribute
                    if hasattr(group_id_encoder, 'classes_'):
                        known_ids = group_id_encoder.classes_
                        print(f"\nThe model knows about {len(known_ids)} group IDs.")
                        print("A sample of known IDs:", known_ids[:10])
                        
                        # Now we check for the problematic ID
                        if '83' in known_ids:
                            print("\nCHECK: '83' IS in the model's known categories.")
                        else:
                            print("\nCHECK: '83' IS NOT in the model's known categories.")
                            
                    else:
                        print("Could not find the '.classes_' attribute on the group_id encoder.")
                else:
                    print("Could not find a 'group_id' encoder in the file.")
            else:
                print("Could not find 'categorical_encoders' in the dataset parameters.")
        else:
            print("Could not find 'dataset_parameters' in the hyperparameters.")
    else:
        print("Could not find 'hyper_parameters' in the checkpoint file.")

except FileNotFoundError:
    print(f"ERROR: Checkpoint file not found at {ckpt_path}")
except Exception as e:
    print(f"An error occurred: {e}")
