from torch import nn
import torch.nn.functional as F

class ImageMultiRegressionModel(nn.Module):
    def __init__(self, model, output_size=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, output_size)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values

