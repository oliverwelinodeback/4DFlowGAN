import numpy as np
from GANetwork.PatchHandler3D import PatchHandler3D
from importlib import import_module

def load_indexes(index_file):
    """Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index"""
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":

    # Module name
    GAN_module_name = "GANetwork"
    data_dir = '../data'

    # ---- Patch index files ----
    training_file = '{}/train.csv'.format(data_dir)  
    validate_file = '{}/validate.csv'.format(data_dir)

    QUICKSAVE = False

    restore = False
    if restore:
        model_dir = '../models/WGAN/GAN-WGAN-1e-3_20250202-1110' 
        model_file = 'GAN-WGAN-epoch100.h5'

    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-4
    epochs = 200
    batch_size = 20
    mask_threshold = 0.5
    epochs_before_disc = 100
    cyclic_gen = None
    cyclic_disc = None

    # Network setting
    folder = 'WGAN'
    network_name = 'GAN-WGAN-1e-3' 
    patch_size = 12
    res_increase = 2 

    #adversarial = 'vanilla'
    #adversarial = 'relativistic'
    adversarial = 'WGAN'

    # Residual blocks (default 4 LR RRDB and 2 HR RRDB)
    low_resblock = 4
    hi_resblock = 2

    # Dynamic import of GAN module's trainer controller
    TrainerController = import_module(GAN_module_name + '.TrainerController', 'src').TrainerController

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # Train set iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None, reduction_factor=1)

    # Validation set iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=True, n_parallel=None, reduction_factor=1)

    # Bechmarking dataset
    testset = None

    # ------- Main Network ------
    print(f"4DFlowGAN Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController(patch_size, res_increase, initial_learning_rate, QUICKSAVE, folder, network_name, low_resblock, hi_resblock, epochs_before_disc, adversarial)
    network.init_model_dir(GAN_module_name)

    print(network.model.summary())
    print(network.generator.summary())
    print(network.discriminator.summary())

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        #network.restore_generator(model_dir, model_file)

        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, cyclic_gen, cyclic_disc, n_epoch=epochs, testset=testset)