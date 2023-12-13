def rename_weights(model_dict):
    # Define the missing and unexpected weights
    missing = [
        "hemisphere.conv.7.conv_res1_bn.0.weight",
        "hemisphere.conv.7.conv_res1_bn.0.bias",
        "hemisphere.conv.7.conv_res1_bn.0.running_mean",
        "hemisphere.conv.7.conv_res1_bn.0.running_var",
        "hemisphere.conv.7.conv_res2_bn.0.weight",
        "hemisphere.conv.7.conv_res2_bn.0.bias",
        "hemisphere.conv.7.conv_res2_bn.0.running_mean",
        "hemisphere.conv.7.conv_res2_bn.0.running_var",
        "hemisphere.conv.16.conv_res1_bn.0.weight",
        "hemisphere.conv.16.conv_res1_bn.0.bias",
        "hemisphere.conv.16.conv_res1_bn.0.running_mean",
        "hemisphere.conv.16.conv_res1_bn.0.running_var",
        "hemisphere.conv.16.conv_res2_bn.0.weight",
        "hemisphere.conv.16.conv_res2_bn.0.bias",
        "hemisphere.conv.16.conv_res2_bn.0.running_mean",
        "hemisphere.conv.16.conv_res2_bn.0.running_var"
    ]

    unexpected = [
        "hemisphere.conv.7.conv_res1_bn.weight",
        "hemisphere.conv.7.conv_res1_bn.bias",
        "hemisphere.conv.7.conv_res1_bn.running_mean",
        "hemisphere.conv.7.conv_res1_bn.running_var",
        "hemisphere.conv.7.conv_res2_bn.weight",
        "hemisphere.conv.7.conv_res2_bn.bias",
        "hemisphere.conv.7.conv_res2_bn.running_mean",
        "hemisphere.conv.7.conv_res2_bn.running_var",
        "hemisphere.conv.16.conv_res1_bn.weight",
        "hemisphere.conv.16.conv_res1_bn.bias",
        "hemisphere.conv.16.conv_res1_bn.running_mean",
        "hemisphere.conv.16.conv_res1_bn.running_var",
        "hemisphere.conv.16.conv_res2_bn.weight",
        "hemisphere.conv.16.conv_res2_bn.bias",
        "hemisphere.conv.16.conv_res2_bn.running_mean",
        "hemisphere.conv.16.conv_res2_bn.running_var",
    ]

    # Create a new state_dict with modified keys
    switched_state_dict = model_dict.copy()
    for missing_key, unexpected_key in zip(missing, unexpected):
        if unexpected_key in switched_state_dict:
            switched_state_dict[missing_key] = switched_state_dict.pop(unexpected_key)

    return switched_state_dict

def rename_weights_bilateral(model_dict):

    missing = [
        "fine_hemi.conv.7.conv_res1_bn.0.weight", 
        "fine_hemi.conv.7.conv_res1_bn.0.bias", 
        "fine_hemi.conv.7.conv_res1_bn.0.running_mean", 
        "fine_hemi.conv.7.conv_res1_bn.0.running_var", 
        "fine_hemi.conv.7.conv_res2_bn.0.weight", 
        "fine_hemi.conv.7.conv_res2_bn.0.bias", 
        "fine_hemi.conv.7.conv_res2_bn.0.running_mean", 
        "fine_hemi.conv.7.conv_res2_bn.0.running_var", 
        "fine_hemi.conv.16.conv_res1_bn.0.weight", 
        "fine_hemi.conv.16.conv_res1_bn.0.bias", 
        "fine_hemi.conv.16.conv_res1_bn.0.running_mean", 
        "fine_hemi.conv.16.conv_res1_bn.0.running_var", 
        "fine_hemi.conv.16.conv_res2_bn.0.weight", 
        "fine_hemi.conv.16.conv_res2_bn.0.bias", 
        "fine_hemi.conv.16.conv_res2_bn.0.running_mean", 
        "fine_hemi.conv.16.conv_res2_bn.0.running_var", 
        "coarse_hemi.conv.7.conv_res1_bn.0.weight", 
        "coarse_hemi.conv.7.conv_res1_bn.0.bias", 
        "coarse_hemi.conv.7.conv_res1_bn.0.running_mean", 
        "coarse_hemi.conv.7.conv_res1_bn.0.running_var", 
        "coarse_hemi.conv.7.conv_res2_bn.0.weight", 
        "coarse_hemi.conv.7.conv_res2_bn.0.bias", 
        "coarse_hemi.conv.7.conv_res2_bn.0.running_mean", 
        "coarse_hemi.conv.7.conv_res2_bn.0.running_var", 
        "coarse_hemi.conv.16.conv_res1_bn.0.weight", 
        "coarse_hemi.conv.16.conv_res1_bn.0.bias", 
        "coarse_hemi.conv.16.conv_res1_bn.0.running_mean", 
        "coarse_hemi.conv.16.conv_res1_bn.0.running_var", 
        "coarse_hemi.conv.16.conv_res2_bn.0.weight", 
        "coarse_hemi.conv.16.conv_res2_bn.0.bias", 
        "coarse_hemi.conv.16.conv_res2_bn.0.running_mean",
        "coarse_hemi.conv.16.conv_res2_bn.0.running_var"]

    unexpected = [
        "fine_hemi.conv.7.conv_res1_bn.weight", 
        "fine_hemi.conv.7.conv_res1_bn.bias", 
        "fine_hemi.conv.7.conv_res1_bn.running_mean", 
        "fine_hemi.conv.7.conv_res1_bn.running_var", 
        "fine_hemi.conv.7.conv_res2_bn.weight", 
        "fine_hemi.conv.7.conv_res2_bn.bias", 
        "fine_hemi.conv.7.conv_res2_bn.running_mean", 
        "fine_hemi.conv.7.conv_res2_bn.running_var", 
        "fine_hemi.conv.16.conv_res1_bn.weight", 
        "fine_hemi.conv.16.conv_res1_bn.bias", 
        "fine_hemi.conv.16.conv_res1_bn.running_mean", 
        "fine_hemi.conv.16.conv_res1_bn.running_var", 
        "fine_hemi.conv.16.conv_res2_bn.weight", 
        "fine_hemi.conv.16.conv_res2_bn.bias", 
        "fine_hemi.conv.16.conv_res2_bn.running_mean", 
        "fine_hemi.conv.16.conv_res2_bn.running_var",
        "coarse_hemi.conv.7.conv_res1_bn.weight", 
        "coarse_hemi.conv.7.conv_res1_bn.bias", 
        "coarse_hemi.conv.7.conv_res1_bn.running_mean", 
        "coarse_hemi.conv.7.conv_res1_bn.running_var", 
        "coarse_hemi.conv.7.conv_res2_bn.weight", 
        "coarse_hemi.conv.7.conv_res2_bn.bias", 
        "coarse_hemi.conv.7.conv_res2_bn.running_mean", 
        "coarse_hemi.conv.7.conv_res2_bn.running_var", 
        "coarse_hemi.conv.16.conv_res1_bn.weight", 
        "coarse_hemi.conv.16.conv_res1_bn.bias", 
        "coarse_hemi.conv.16.conv_res1_bn.running_mean", 
        "coarse_hemi.conv.16.conv_res1_bn.running_var", 
        "coarse_hemi.conv.16.conv_res2_bn.weight", 
        "coarse_hemi.conv.16.conv_res2_bn.bias", 
        "coarse_hemi.conv.16.conv_res2_bn.running_mean", 
        "coarse_hemi.conv.16.conv_res2_bn.running_var"]

    # Create a new state_dict with modified keys
    switched_state_dict = model_dict.copy()
    for missing_key, unexpected_key in zip(missing, unexpected):
        if unexpected_key in switched_state_dict:
            switched_state_dict[missing_key] = switched_state_dict.pop(unexpected_key)

    return switched_state_dict
