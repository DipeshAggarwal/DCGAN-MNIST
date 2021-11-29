import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU

class DCGAN(keras.Model):
    
    def __init__(self, generator, discriminator, latent_dim):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Train the discriminator with both real and fake images
        with tf.GradientTape() as tape:
            # Compute discrimantor loss on real images
            pred_real = self.discriminator(real_images, training=True)
            d_loss_real = self.loss_fn(tf.ones((batch_size, 1)), pred_real)
            
            # Compute discrimantor loss on fake images
            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images, training=True)
            d_loss_fake = self.loss_fn(tf.zeros((batch_size, 1)), pred_fake)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            
        # Compute discrimantor gradient
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Update discrimantor weights
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        # Train the generator (do not update weights of the discriminator)
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise)
            pred_fakes = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fakes)
            
        # Compute generator gradients
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Update generator weights
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
            }
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

def build_generator(width, height, depth, latent_dim):
     # Weight init. Taken from DCGAN paper
    WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    w = width // 4
    h = height // 4
    
    model = Sequential(name="Generator")
    
    model.add(Dense(w * h * 256, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((w, h, 256)))
    
    # Upsmaple the image to half of width and height
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    # Upsample the image to width and height
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(Conv2D(depth, (5, 5), padding="same", activation="tanh"))
    
    return model
    
def build_discriminator(width, height, depth, alpha=0.2):
    model = Sequential(name="Discrimantor")
    input_shape = (height, width, depth)
    
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation="sigmoid"))
    
    return model