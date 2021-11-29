import os

OUTPUT = "output"
VIS_DIR = os.path.sep.join([OUTPUT, "epoch_vis", ""])
MODEL_PATH = os.path.sep.join([OUTPUT, "generator.h5"])

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

WIDTH = 28
HEIGHT = 28
DEPTH = 1

LATENT_DIM = 100
INIT_LR = 0.0002
EPOCHS = 50
