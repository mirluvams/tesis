import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import torchvision.transforms.v2 as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, PreTrainedModel, PretrainedConfig, AutoConfig
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
    "microsoft/beit-base-patch16-224",
    "google/vit-base-patch16-224",
]

image_size=224
model_base=model_names[3]
model_output_name="vit-base"
model_output=f"./models/{model_output_name}"
model_output_tmp=f"~/tmp/models/{model_output_name}"

try:
    model=AutoModelForImageClassification.from_pretrained(model_output)
except:
    model=AutoModelForImageClassification.from_pretrained(model_base)


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
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    report_to=[]
)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])#.to(DEVICE)
    labels = torch.tensor([example["class"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels, }

def compute_metrics(pred):
    x,y=np.argmax(pred.predictions, axis=1), pred.label_ids
    #print(x,y)
    return {"accuracy":sklearn.metrics.accuracy_score(x, y),
           "f1":sklearn.metrics.f1_score(x,y, average='weighted')}

trainer = Trainer(
    model,
    trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)
print(trainer.evaluate())
print(trainer.train())

#trainer.save_metrics("all", )
trainer.save_model(model_output)
trainer.save_state()
