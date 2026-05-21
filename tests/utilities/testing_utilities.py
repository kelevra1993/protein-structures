import torch
from torch import nn
from pathlib import Path
from typing import List, Dict


def test_nn_module_method(module: nn.Module, input_tensor_dictionary: Dict[str, torch.Tensor],
                          output_tensor_names: List[str],
                          reference_folder: str | Path, batch_size: tuple[int]):
    # First We Just Modify The Parameters Of The Module To Be Fixed
    # Set To eval() To Disable Dropout Or Any Batch Normalisation (anything non deterministic)
    module.eval()
    module.double()

    with torch.no_grad():
        for param in module.parameters():
            # Set to the same parameters.
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

    # Module Is Now Ready For Testing

    # Simple Input And Batched Input
    simple_input_dictionary = {input_name: input_tensor for input_name, input_tensor in input_tensor_dictionary.items()}
    batched_input_dictionary = {input_name: input_tensor.broadcast_to((batch_size,) + input_tensor.shape)
                                for input_name, input_tensor in input_tensor_dictionary.items()}

    # Simple Output And Batched Output
    simple_output = module(**simple_input_dictionary)
    batched_output = module(**batched_input_dictionary)

    out_file_names = [f'{reference_folder}/{out_name}.pt' for out_name in output_tensor_names]

    for output_tensor, batched_output_tensor, output_filename, output_tensor_name in zip(simple_output, batched_output,
                                                                                         out_file_names,
                                                                                         output_tensor_names):
        expected_output_tensor = torch.load(output_filename, weights_only=True)

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'

        expected_out_batch_shape = (batch_size,) + expected_output_tensor.shape
        expected_batched_output_tensor = expected_output_tensor.unsqueeze(0).broadcast_to(expected_out_batch_shape)

        assert torch.allclose(batched_output_tensor, expected_batched_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'
