import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import torchvision.transforms.v2 as transforms
from transformers import AutoImageProcessor, AutoModel, TrainingArguments, Trainer, PreTrainedModel, PretrainedConfig, AutoConfig
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
    "microsoft/swinv2-base-patch4-window16-256",
    "facebook/dinov2-base",
    "microsoft/beit-base-patch16-224",
    "google/vit-base-patch16-224",
]

image_size=256
model_base=model_names[0]
model_output_name="swinv2-tiny"

class ImageMultiRegressionConfig(PretrainedConfig):
    def __init__(
        self,
        output_size=3,
        init_checkpoint=None,
        **kwargs,
    ):
        self.output_size=output_size
        self.init_checkpoint=init_checkpoint
        super().__init__(**kwargs)


class ImageMultiRegressionModel(PreTrainedModel):
    config_class=ImageMultiRegressionConfig
    def __init__(self, config, loss=nn.MSELoss()):
        super().__init__(config)
        
        self.inner_model = AutoModel.from_pretrained(config.init_checkpoint)
        self.classifier = nn.Linear(self.inner_model.config.hidden_size, config.output_size)
        self.loss=loss
    
    def forward(self, pixel_values, labels=None):
        outputs = self.inner_model(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # image embedding
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = self.loss(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values


#image_processor = AutoImageProcessor.from_pretrained(model_name)
try:
    model=ImageMultiRegressionModel.from_pretrained(model_output)
except:
    print("INITIALIZING NEW MODEL AS PRETRAINED LOAD FAILED")
    config=ImageMultiRegressionConfig(output_size=3, init_checkpoint=model_base)
    model=ImageMultiRegressionModel(config)


#model_name="microsoft/swinv2-base-patch4-window16-256"
dataset=load_from_disk("./data/dataset/")
dataset["train"].set_format("torch")
dataset["test"].set_format("torch")
_train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(image_size,image_size), scale=(.6,1.0), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_transform(ex):
    ex["pixel_values"]=[_train_transform(image) for image in ex["image"]]
    return ex
    
_test_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    transforms.Resize(size=(image_size,image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def test_transform(ex):
    ex["pixel_values"]=[_test_transform(image) for image in ex["image"]]
    return ex

#_columns=["pixel_values", "light_level", "fume_strength", "explosion_strength"]
dataset["train"].set_transform(train_transform)
dataset["test"].set_transform(test_transform)

os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"


model_output_temp=f"~/tmp/models/{model_output_name}/"

trainer_args = TrainingArguments(
    model_output_temp,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_strategy="epoch",
    #learning_rate=5e-5,
    fp16=True,
    per_device_train_batch_size=8,
    #gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=25,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    metric_for_best_model="R2_test",
    greater_is_better=True,
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
           "R2_test":sklearn.metrics.r2_score(x,y)}

trainer = Trainer(
    model,
    trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

print(trainer.train())

#trainer.save_metrics("all", )
trainer.save_model(f"models/{model_output_name}")
trainer.save_state()
