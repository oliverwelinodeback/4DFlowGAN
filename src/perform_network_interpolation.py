import tensorflow as tf
import numpy as np
import os
from importlib import import_module

def prepare_network(SR4DFlowGAN, patch_size, res_increase, low_resblock, hi_resblock):

    # Network & output
    net = SR4DFlowGAN(patch_size, res_increase)
    generator = net.build_generator(low_resblock, hi_resblock, channel_nr=36)
    discriminator = net.build_WGAN_discriminator(channel_nr=32)
    model = net.build_network(generator, discriminator)

    return model, generator

def interpolate_models(model_psnr, model_gan, alpha):
    """
    Interpolate between two models based on alpha.
    model_psnr: Generator model trained for PSNR.
    model_gan: Generator model trained with GAN.
    alpha: Interpolation factor.
    """

    # Retrieve weights from both models
    psnr_weights = model_psnr.get_weights()
    gan_weights = model_gan.get_weights()

    # Check if the number of weight sets matches
    if len(psnr_weights) != len(gan_weights):
        raise ValueError("The models have a different number of layers")

    # Calculate interpolated weights
    interpolated_weights = [(1 - alpha) * np.array(pw) + alpha * np.array(gw) 
                            for pw, gw in zip(psnr_weights, gan_weights)]

    # Create a new model based on the model_psnr's architecture
    interpolated_model = tf.keras.models.clone_model(model_psnr)
    interpolated_model.set_weights(interpolated_weights)

    return interpolated_model

def print_model_weights(model, layer_indexes=[0], weight_indexes=[0]):
    """
    Print weights for specified layers and indexes.
    
    Parameters:
    model (tf.keras.Model): The model from which to print weights.
    layer_indexes (list of int): Indices of the layers from which to print weights.
    weight_indexes (list of int): Indices of the weights to print from the selected layers.
    """
    weights = model.get_weights()
    for layer_idx in layer_indexes:
        layer_weights = weights[layer_idx]
        print(f"Weights from layer {layer_idx}:")
        for weight_idx in weight_indexes:
            print(f"  Weight at index {weight_idx}: {layer_weights.flatten()[weight_idx]}")

if __name__ == '__main__':

    # NOTE
    # All architecure parameters must match those used during training for the stored weights to be loaded successfully.
    # Relevant options are: patch size, res increase, low/hi resblocks, gen/disc channel count
    # If unsure about the values used, check the backup_source of the saved model.

    ### ---------------- SETTINGS ----------------

    GAN_module_name = "GANetwork"
    model_path_psnr = "../models/4DFlowGen/4DFlowGen-epoch100.h5" 
    model_path_gan = "../models/WGAN/GAN-WGAN-best.h5"

    output_dir = "../models/interpolated_models"

    output_name = "GAN-WGAN_1e-4_alpha1"
    alpha = 1.0

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Params
    patch_size = 12
    res_increase = 2

    # Network
    low_resblock=4
    hi_resblock=2

    ### ------------------------------------------------

    # Dynamic import of GAN module's architecture 
    SR4DFlowGAN = import_module(GAN_module_name + '.SR4DFlowGAN', 'src').SR4DFlowGAN

    print(f"Loading {GAN_module_name}")
    
    # Load the network
    network_psnr, generator = prepare_network(SR4DFlowGAN, patch_size, res_increase, low_resblock, hi_resblock)
    network_psnr.load_weights(model_path_psnr)
    print("PSNR:")
    print(network_psnr.summary())

    network_gan, generator = prepare_network(SR4DFlowGAN, patch_size, res_increase, low_resblock, hi_resblock)
    network_gan.load_weights(model_path_gan)
    print("GAN:")
    print(network_gan.summary())

    network_interpolated = interpolate_models(network_psnr, network_gan, alpha)
    print("Interpolated:")
    print(network_interpolated.summary())

    network_interpolated.save(f'{output_dir}/{output_name}.h5')

    # Example usage: print weights from the first layer, first weight
    # print_model_weights(network_gan, [0], [0])
    # print_model_weights(network_psnr, [0], [0])
    # print_model_weights(network_interpolated, [0], [0])