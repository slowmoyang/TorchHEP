from torch import Tensor
import torch.nn as nn

class Objwise(nn.Module):
    r"""
    """

    def __init__(self, *operations: nn.Module) -> None:
        super().__init__()
        assert len(operations) >= 1
        self.operations = nn.Sequential(*operations)

    def forward(self,
                input: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        batch_size, length, num_features = input.shape

        select_mask = data_mask.reshape(batch_size * length, 1)

        input = input.reshape(-1, num_features)
        input = input.masked_select(select_mask)
        input = input.reshape(-1, num_features)

        output_source = self.operations(input)
        output_size = output_source.size(1)

        scatter_mask = select_mask.expand(select_mask.size(0), output_size)
        output = input.new_zeros((batch_size * length, output_size))
        output = output.masked_scatter(mask=scatter_mask,
                                       source=output_source)
        output = output.reshape(batch_size, length, output_size)

        return output

