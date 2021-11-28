import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt

class GANMonitor(Callback):
    
    def __init__(self, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim
        
        # Create random seed for visualisation during training
        self.seed = tf.random.normal([16, latent_dim])
        
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.seed)
        generated_images = (generated_images - 127.5) / 127.5
        generated_images.numpy()
        
        # Create a num_size * num_size plot of images
        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            img = tf.keras.utils.array_to_img(generated_images[i])
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            
        # Save the image
        plt.savefig("epoch_{:03d}.png".format(epoch))
        plt.show()
        
    def on_train_end(self, logs=None):
        self.model.generator.save("generator.h5")
        