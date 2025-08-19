from pytorch_forecasting import TimeSeriesDataSet
import torch
from typing import Dict, List, Tuple
from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models import BaseModelWithCovariates
from torch import nn


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x)



class TFTwithCovariates(BaseModelWithCovariates):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        n_hidden_layers: int,
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals


        **kwargs
    ):


        self.save_hyperparameters()
        super().__init__(**kwargs)

        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        network_input = x["encoder_cont"].squeeze(-1)
        prediction=self.network(network_input)
        # rescale predictions into target space
        prediction=self.transform_output(prediction, target_scale=x["target_scale"])

        return self.to_network_output(prediction=prediction)
    
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }

        new_kwargs.update(kwargs)

        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset._time_varying_known_categoricals) == 0
            and len(dataset._time_varying_known_reals) == 0
            and len(dataset._time_varying_unknown_categoricals) == 0
            and len(dataset._static_categoricals) == 0
            and len(dataset._static_reals) == 0
            and len(dataset._time_varying_unknown_reals) == 1
            and dataset._time_varying_unknown_reals[0] == dataset.target
        ), "Only covariate should be the target in 'time_varying_unknown_reals'"
        return super().from_dataset(dataset, **new_kwargs)

