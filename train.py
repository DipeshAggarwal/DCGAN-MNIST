from core import config
from core.dcgan import DCGAN
from core.dcgan import build_discriminator
from core.dcgan import build_generator
from core.callbacks import GANMonitor
from core.helpers import info
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

info("Downnload and split model.")
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

info("Reshape the dataset to add a channel dimension and convert it to float.")
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")

info("Normalise the value in the range [-1, 1] because we use tanh in generator.")
train_images = (train_images - 127.5) / 127.5

# print(build_generator(28, 28, 1).summary())
# print(build_discriminator(28, 28, 1).summary())

discriminator = build_discriminator(width=config.WIDTH, height=config.HEIGHT, depth=config.DEPTH)
generator = build_generator(width=config.WIDTH, height=config.HEIGHT, depth=config.DEPTH, latent_dim=config.LATENT_DIM)

info("Compiling the model.")
dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=config.LATENT_DIM)
dcgan.compile(d_optimizer=Adam(lr=config.INIT_LR, beta_1=0.5),
              g_optimizer=Adam(lr=config.INIT_LR, beta_1=0.5),
              loss_fn=BinaryCrossentropy()
              )

info("Training the model.")
dcgan.fit(
    train_images,
    epochs=config.EPOCHS,
    callbacks=[GANMonitor(num_img=16, latent_dim=config.LATENT_DIM, vis_path=config.VIS_DIR, model_path=config.MODEL_PATH)])
