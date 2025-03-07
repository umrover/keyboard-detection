import pickle
from ultralytics import YOLO

from keyrover import *
from keyrover.vision import *
from keyrover.images.texcoord import *
from keyrover.images.key_mask import *
from keyrover.color import NamedPalette
from keyrover.vision.models import CornersRegressionModel
from keyrover.datasets import KeyboardCornersDataset
import torch
import matplotlib.pyplot as plt

SIZE = (256, 256)

dataset = "v4-nodistort"
test_dataset = KeyboardCornersDataset([], size=SIZE, version=dataset)

model = CornersRegressionModel.load("magic-wave-28.pt")

torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), 'magic.onnx', opset_version=12)
