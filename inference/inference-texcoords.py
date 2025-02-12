from torchvision.transforms import v2 as transforms

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from keyrover import *
from keyrover.vision import *
from keyrover.vision.models import TexcoordsRegressionModel
from keyrover.datasets import KeyboardTexcoordDataset

import wandb

wandb.login()
device

SIZE = (256, 256)

train_dataset, valid_dataset, test_dataset = KeyboardTexcoordDataset.load("v4", size=SIZE, kind="tanh")
len(train_dataset), len(valid_dataset), len(test_dataset)
train_dataset.set_transforms([
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomAffine(degrees=30, shear=30, translate=(0.25, 0.25)),
    # transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
])

train_dataset.set_input_augmentations([
    # transforms.GaussianNoise(sigma=0.1, clip=True),
    # transforms.RandomApply([transforms.GaussianNoise(sigma=0.1, clip=False)], p=0.5),
], norm_params="default")

img, mask = train_dataset.random_img()
imshow(img, mask)

describe(mask)
describe(img)

train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(train_dataset, valid_dataset, test_dataset,
                                                                         batch_size=48, num_workers=2,
                                                                         persistent_workers=True, pin_memory=True)
ARCH = "unet"
BACKBONE = "timm-regnety_002"

LEARNING_RATE = 2e-3

wandb.finish()
model = TexcoordsRegressionModel(ARCH, BACKBONE, in_channels=3, out_classes=2, encoder_weights="imagenet",
                                 lr=LEARNING_RATE, activation="tanh")
model

summarize(model)

logger = WandbLogger(project="mrover-keyboard-texcoords-segmentation")

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

trainer = pl.Trainer(log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

model.save(f"models/texcoords/{wandb.run.name}-{ARCH}-{BACKBONE}.pt")
img, mask = valid_dataset.random_img()
pred = model.predict(img)

imshow(img, pred)
imshow((mask - pred) ** 2, mask)
