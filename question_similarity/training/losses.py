from typing import Iterable

from sentence_transformers import SentenceTransformer
from torch import nn, Tensor
import torch.nn.functional as F


class Loss(nn.Module):
    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        raise NotImplementedError(
            "Loss is an abstract class. Implement forward method."
        )


class ContrastiveLoss(Loss):
    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.margin = margin

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        assert (
            len(sentence_features) == 2
        ), "ContrastiveLoss requires exactly two question features"
        question1 = self.model(sentence_features[0])["sentence_embedding"]
        question2 = self.model(sentence_features[1])["sentence_embedding"]
        distances = 1 - F.cosine_similarity(question1, question2)
        losses = 0.5 * (
            labels.float() * distances.pow(2)
            + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean()
