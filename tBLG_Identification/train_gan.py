# train.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from evaluation import get_rmse_on_batch, get_prmsd_on_batch, get_mae_on_batch

numofEPOCHS = 20000

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(data, generator, discriminator, noise_dim):
    """
    Function for implementing one training step of the GAN model
    """
    noise = tf.random.normal([data.shape[0], noise_dim], seed=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs, generator, discriminator, noise_dim, checkpoint, checkpoint_prefix):
    """
    Main GAN Training Function
    """
    epochs_gen_losses, epochs_disc_losses, epochs_accuracies = [], [], []

    for epoch in range(epochs):

        gen_losses, disc_losses = [], []

        for data_batch in dataset:
            gen_loss, disc_loss = train_step(data_batch, generator, discriminator, noise_dim)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        epoch_gen_loss = np.average(gen_losses)
        epoch_disc_loss = np.average(disc_losses)
        epochs_gen_losses.append(epoch_gen_loss)
        epochs_disc_losses.append(epoch_disc_loss)

        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Generator Loss: {}, Discriminator Loss: {}".format(epoch_gen_loss, epoch_disc_loss))


        # Draw the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            draw_training_evolution(generator, epoch+1, tf.random.normal([1,noise_dim]))

        # Save the model every 5 epochs for the last 5000 epochs
        if (epoch + 1) % 5 == 0 and epoch > (numofEPOCHS - 5000):
            checkpoint.save(file_prefix=checkpoint_prefix)  # Comment not to save model checkpoints while training

    return epochs_gen_losses, epochs_disc_losses, epochs_accuracies

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

# Defining the generator loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def draw_training_evolution(model, epoch, noise_input):
  """
    Function that takes in the generator model, epoch number, and
    does a prediction and plots the generated singal then saves it.
  """
  # `training` is set to False.
  # Thus, all layers run in inference mode (batchnorm).
  predictions = model(noise_input, training=False)

  for i in range(predictions.shape[0]):
    fig = plt.figure()
    plt.plot(predictions[i].numpy().tolist())
    plt.savefig('output/process/image_at_epoch_{:04d}_{:04d}.png'.format(epoch, i))
    plt.close()