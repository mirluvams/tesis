import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import torchvision.transforms.v2 as transforms
from transformers import AutoImageProcessor, AutoModel, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk
from evaluate import load as load_metric
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import os

# pick one, then switch to the local model on retrains.
model_names=[
    "microsoft/swinv2-tiny-patch4-window16-256",
    "facebook/dinov2-base",
    "nvidia/MambaVision-B-1K",
    "microsoft/beit-base-patch16-224",
    "microsoft/swinv2-base-patch4-window16-256",
    "google/vit-base-patch16-384",
]

model_name=model_names[0]
model_output="~/tmp/models/swinv2-tiny/"



class ImageMultiRegressionModel(nn.Module):
    def __init__(self, model, loss=nn.MSELoss(), output_size=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, output_size)
        self.loss=loss
    
    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # image embedding
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = self.loss(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values
    

#model_name="microsoft/swinv2-base-patch4-window16-256"
dataset=load_from_disk("./data/dataset/")
dataset["train"].set_format("torch")
dataset["test"].set_format("torch")
_train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(256,256), scale=(.6,1.0), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_transform(ex):
    ex["pixel_values"]=[_train_transform(image) for image in ex["image"]]
    return ex
    
_test_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    transforms.Resize(size=(256,256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def test_transform(ex):
    ex["pixel_values"]=[_test_transform(image) for image in ex["image"]]
    return ex

#_columns=["pixel_values", "light_level", "fume_strength", "explosion_strength"]
dataset["train"].set_transform(train_transform)
dataset["test"].set_transform(test_transform)

image_processor = AutoImageProcessor.from_pretrained(model_name)
raw_model=AutoModel.from_pretrained(model_name)
model=ImageMultiRegressionModel(raw_model, output_size=3)

os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

trainer_args = TrainingArguments(
    model_output,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    #learning_rate=5e-5,
    fp16=True,
    per_device_train_batch_size=12,
    #gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=25,
    warmup_ratio=0.1,
    logging_steps=25,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    #metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to=[]
)


_columns=["light_level","fume_strength","explosion_strength"]
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])#.to(DEVICE)
    labels =  torch.stack([torch.tensor([example[c] for c in _columns]) for example in examples])#.to(DEVICE)
    return {"pixel_values": pixel_values, "labels": labels, }

def compute_metrics(pred):
    x,y=pred.predictions, pred.label_ids
    return {"MSE":sklearn.metrics.mean_squared_error(x, y),
           "MAE":sklearn.metrics.mean_absolute_error(x,y),
           "R2":sklearn.metrics.r2_score(x,y)}

trainer = Trainer(
    model,
    trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

print(trainer.train())

