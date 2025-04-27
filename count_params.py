import torch

def count_parameters_in_pth(pth_path):
    """
    Count the number of parameters in a PyTorch .pth file.

    Args:
        pth_path (str): Path to the .pth file.

    Returns:
        int: Total number of parameters.
    """
    state_dict = torch.load(pth_path, map_location='cpu')

    # If it's a full checkpoint, get the 'state_dict' inside it
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    total_params = sum(param.numel() for param in state_dict.values())
    return total_params

print(count_parameters_in_pth("checkpoints/final_weights.pth"))