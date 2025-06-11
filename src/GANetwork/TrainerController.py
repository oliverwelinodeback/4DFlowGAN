import pickle
import tensorflow as tf
import numpy as np
import datetime
import time
import shutil
import os
from .SR4DFlowGAN import SR4DFlowGAN
from skimage import morphology
from GANetwork import utility, h5util, loss_utils

class TrainerController:
    # constructor
    def __init__(self, patch_size, res_increase, initial_learning_rate=1e-4, quicksave_enable=True, network_folder='runs', network_name='4DFlowGAN', lr_resblock=4, hr_resblock=2, epochs_before_disc=100, adversarial='vanilla'):
        """
            TrainerController constructor
            Setup all the placeholders, network graph, loss functions and optimizer here.
        """
        # Loss weights
        self.non_fluid_weight = 1.0 # Weighting non-fluid region
        self.boundary_weight = 1.0 # Weighting boundary region
        self.core_weight = 1.0 # Weighting core region
        
        # Discriminator loss weights
        self.discL2weight = 100
        self.disc_loss_weight = 1e-3 

        self.adversarial = adversarial
        self.mask_for_disc = True # Apply mask to SR field before being passed to the discriminator

        # General parameters
        self.epochs_before_disc = epochs_before_disc
        self.res_increase = res_increase
        self.QUICKSAVE_ENABLED = quicksave_enable
        
        # Network
        self.network_folder = network_folder
        self.network_name = network_name

        net = SR4DFlowGAN(patch_size, res_increase)
        self.generator = net.build_generator(lr_resblock, hr_resblock, channel_nr=36)

        if self.adversarial in ['relativistic', 'WGAN']:
            self.discriminator = net.build_discriminator(channel_nr=32, logits=True)
        else:
            self.discriminator = net.build_discriminator(channel_nr=32, logits=False)

        self.model = net.build_network(self.generator, self.discriminator)

        # ===== Metrics =====
        self.loss_metrics = dict([
            ('train_accuracy', tf.keras.metrics.Mean(name='train_accuracy')),
            ('val_accuracy', tf.keras.metrics.Mean(name='val_accuracy')),
            ('train_gen_loss', tf.keras.metrics.Mean(name='train_gen_loss')),
            ('val_gen_loss', tf.keras.metrics.Mean(name='val_gen_loss')),
            ('train_mse_fluid', tf.keras.metrics.Mean(name='train_mse_fluid')),
            ('val_mse_fluid', tf.keras.metrics.Mean(name='val_mse_fluid')),
            ('train_mse_boundary', tf.keras.metrics.Mean(name='train_mse_boundary')),
            ('val_mse_boundary', tf.keras.metrics.Mean(name='val_mse_boundary')),
            ('train_rmse_boundary', tf.keras.metrics.Mean(name='train_rmse_boundary')),
            ('val_rmse_boundary', tf.keras.metrics.Mean(name='val_rmse_boundary')),
            ('train_mse_core', tf.keras.metrics.Mean(name='train_mse_core')),
            ('val_mse_core', tf.keras.metrics.Mean(name='val_mse_core')),
            ('train_mse_nf', tf.keras.metrics.Mean(name='train_mse_nf')),
            ('val_mse_nf', tf.keras.metrics.Mean(name='val_mse_nf')),

            ('train_gen_adv_loss', tf.keras.metrics.Mean(name='train_gen_adv_loss')),
            ('val_gen_adv_loss', tf.keras.metrics.Mean(name='val_gen_adv_loss')),
            ('train_disc_loss', tf.keras.metrics.Mean(name='train_disc_loss')),
            ('val_disc_loss', tf.keras.metrics.Mean(name='val_disc_loss')),

            ('train_wgan_gradpen', tf.keras.metrics.Mean(name='train_wgan_gradpen')),
            ('val_wgan_gradpen', tf.keras.metrics.Mean(name='val_wgan_gradpen')),

            ('l2_gen_loss', tf.keras.metrics.Mean(name='l2_gen_loss')),
            ('l2_disc_loss', tf.keras.metrics.Mean(name='l2_disc_loss')),

        ])
        self.accuracy_metric = 'val_accuracy'
        print(f"Accuracy metric: {self.accuracy_metric}")

        # Learning rate and optimizer
        self.learning_rate = initial_learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model to save optimizer weights
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def get_core_mask(self, mask):
        return morphology.binary_erosion(mask)

    def loss_function(self, y_true, y_pred, mask, compartment):
        """
            Calculate Total Loss function
        """
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        # Calculate voxel-wise error
        mse = self.calculate_mse(u,v,w, u_pred,v_pred,w_pred)

        # Extract boundary, core, and non-fluid mask
        core_mask = tf.numpy_function(self.get_core_mask, [mask], tf.bool)
        core_mask = tf.cast(core_mask, dtype=tf.float32)
        core_mask = tf.ensure_shape(core_mask, mask.shape)

        boundary_mask = mask - core_mask
        boundary_mask = tf.cast(boundary_mask, dtype=tf.float32)
        boundary_mask = tf.ensure_shape(boundary_mask, mask.shape)

        non_fluid_mask = tf.less(mask, tf.constant(0.5))
        non_fluid_mask = tf.cast(non_fluid_mask, dtype=tf.float32)
        non_fluid_mask = tf.ensure_shape(non_fluid_mask, mask.shape)

        epsilon = 1 # minimum 1 pixel

        # Fluid error
        fluid_mse = mse * mask
        fluid_mse = tf.reduce_sum(fluid_mse, axis=[1,2,3]) / (tf.reduce_sum(mask, axis=[1,2,3]) + epsilon)

        # Core error
        core_mse = mse * core_mask
        core_mse = tf.reduce_sum(core_mse, axis=[1,2,3]) / (tf.reduce_sum(core_mask, axis=[1,2,3]) + epsilon)

        # Boundary error
        boundary_mse = mse * boundary_mask
        boundary_mse = tf.reduce_sum(boundary_mse, axis=[1,2,3]) / (tf.reduce_sum(boundary_mask, axis=[1,2,3]) + epsilon)
        boundary_rmse = tf.sqrt(boundary_mse)

        # Non fluid error
        non_fluid_mse = mse * non_fluid_mask
        non_fluid_mse = tf.reduce_sum(non_fluid_mse, axis=[1,2,3]) / (tf.reduce_sum(non_fluid_mask, axis=[1,2,3]) + epsilon)

        # Total error
        mse = boundary_mse * self.boundary_weight + core_mse * self.core_weight + non_fluid_mse * self.non_fluid_weight
        total_loss = tf.reduce_mean(mse)
        total_loss = mse

        # Return all losses for logging
        return  total_loss, mse, fluid_mse, non_fluid_mse, boundary_mse, boundary_rmse, core_mse

    def calculate_regularizer_loss(self, net=None):
        """
            https://stackoverflow.com/questions/62440162/how-do-i-take-l1-and-l2-regularizers-into-account-in-tensorflow-custom-training
        """
        loss = 0
        if net is None:
            net = self.model
        for l in net.layers:
            # if hasattr(l,'layers') and l.layers: # the layer itself is a model
            #     loss+=add_model_loss(l)
            if hasattr(l,'kernel_regularizer') and l.kernel_regularizer:
                loss+=l.kernel_regularizer(l.kernel)
            if hasattr(l,'bias_regularizer') and l.bias_regularizer:
                loss+=l.bias_regularizer(l.bias)
        return loss

    def accuracy_function(self, y_true, y_pred, mask):
        """
            Calculate relative speed error
        """
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        return loss_utils.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

    def calculate_mse(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate Speed magnitude error (squared)
        """
        return (u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2
    
    def calculate_mae(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate Speed magnitude error (absolute values)
        """
        return tf.abs(u_pred - u) +  tf.abs(v_pred - v) + tf.abs(w_pred - w)
    
    def calculate_huber_loss(self, u, v, w, u_pred, v_pred, w_pred, delta = 0.01):
        """
            Calculate Speed magnitude error (absolute values)
        """
        huber_mse = 0.5*((u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2)
        huber_mae = delta * (tf.abs(u_pred - u) + tf.abs(v_pred - v) + tf.abs(w_pred - w) - 0.5 * delta)

        return tf.where(tf.abs(u_pred - u) + tf.abs(v_pred - v) + tf.abs(w_pred - w) <= delta, huber_mse, huber_mae)

    def calculate_bce(self, target, pred):
        """
            Calculate Binary Cross Entropy
        """
        return tf.keras.losses.BinaryCrossentropy()(target, pred)

    def init_model_dir(self, GAN_module):
        """
            Create model directory to save the weights with a [network_folder]/[network_name]_[datetime] format
            Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"../models/{self.network_folder}/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        # summary - Tensorboard stuff
        self._prepare_logfile_and_summary(GAN_module)
    
    def _prepare_logfile_and_summary(self, GAN_module):
        """
            Prepare csv logfile to keep track of the loss and Tensorboard summaries
        """
        # summary - Tensorboard stuff
        self.train_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/train')
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/validate')

        # Prepare log file
        self.logfile = self.model_dir + '/loss.csv'

        utility.log_to_file(self.logfile, f'Network: {self.network_name}\n')
        utility.log_to_file(self.logfile, f'Initial learning rate: {self.learning_rate}\n')
        utility.log_to_file(self.logfile, f'Accuracy metric: {self.accuracy_metric}\n')

        # Header
        stat_names = ','.join(self.loss_metrics.keys()) # train and val stat names
        utility.log_to_file(self.logfile, f'epoch, {stat_names}, learning rate, elapsed (sec), best_model, saved_model\n')

        print("Copying source code to model directory...")
        # Copy all the source file to the model dir for backup
        directory_to_backup = [".", "GANetwork", GAN_module]
        for directory in directory_to_backup:
            files = os.listdir(directory)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir,"backup_source",directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)

                    shutil.copy2(f"{directory}/{fname}", dest_fpath)
      
    #@tf.function
    def train_step(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs

        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        with tf.GradientTape() as gen_tape:
            input_data = tf.concat([u,v,w], axis=-1)
            predictions = self.generator(input_data, training=True)

            gen_loss = self.gen_calculate_and_update_metrics(hires, predictions, mask, compartment, 'train')

        # Get the gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        
        # Update the weights
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    #@tf.function
    def train_step_relativistic(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs

        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        with tf.GradientTape() as gen_tape:
            input_data = tf.concat([u,v,w], axis=-1)
            
            x_pred = self.generator(input_data, training=True)
            x_pred_p = x_pred * tf.expand_dims(mask, -1) if self.mask_for_disc else x_pred

            self.discriminator.trainable = True
            with tf.GradientTape() as disc_tape:
                real_y_pred = self.discriminator(x_target, training=True)
                fake_y_pred = self.discriminator(x_pred_p, training=True)

                disc_loss = self.disc_calculate_and_update_metrics_relativistic(fake_y_pred, real_y_pred, 'train')

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            self.discriminator.trainable = False
            
            gen_loss = self.gen_calculate_and_update_metrics_relativistic(x_target, x_pred, real_y_pred, fake_y_pred, mask, 'train', compartment)

        # Get the gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    #@tf.function
    def train_step_vanilla(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs
        
        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        with tf.GradientTape() as gen_tape:
            input_data = tf.concat([u,v,w], axis=-1)
            x_pred = self.generator(input_data, training=True)
            x_pred_p = x_pred * tf.expand_dims(mask, -1) if self.mask_for_disc else x_pred

            self.discriminator.trainable = True
            with tf.GradientTape() as disc_tape:
                real_y_pred = self.discriminator(x_target, training=True)
                fake_y_pred = self.discriminator(x_pred_p, training=True)
                real_y_target = np.ones(fake_y_pred.shape)
                fake_y_target = np.zeros(fake_y_pred.shape)

                disc_loss = self.disc_calculate_and_update_metrics_vanilla(fake_y_target, fake_y_pred, real_y_target, real_y_pred, 'train')

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            self.discriminator.trainable = False

            gen_loss = self.gen_calculate_and_update_metrics_vanilla(x_target, x_pred, real_y_target, fake_y_pred, mask, 'train', compartment)

        # Get the gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    @tf.function
    def train_step_WGAN(self, data_pairs, lambda_gp=10):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs
        
        x_real = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        with tf.GradientTape() as gen_tape:
            input_data = tf.concat([u,v,w], axis=-1)
            x_fake = self.generator(input_data, training=True)
            x_fake_masked = x_fake * tf.expand_dims(mask, -1) if self.mask_for_disc else x_fake
            
            self.discriminator.trainable = True
            with tf.GradientTape() as disc_tape:
                real_y_score = self.discriminator(x_real, training=True)
                fake_y_score = self.discriminator(x_fake_masked, training=True)

                w_distance = tf.reduce_mean(real_y_score) - tf.reduce_mean(fake_y_score)

                l2_reg_loss = self.calculate_regularizer_loss(self.discriminator) * self.discL2weight

                # Gradient Penalty Calculation
                grad_penalty = self.calculate_gradient_penalty(x_real, x_fake_masked, lambda_gp)
                disc_loss = -w_distance + grad_penalty + l2_reg_loss

            self.loss_metrics[f'l2_disc_loss'].update_state(l2_reg_loss)
            self.loss_metrics[f'train_disc_loss'].update_state(disc_loss)
            self.loss_metrics[f'train_wgan_gradpen'].update_state(grad_penalty)

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            self.discriminator.trainable = False

            # Generator Training
            gen_loss = self.gen_calculate_and_update_metrics_WGAN(x_real, x_fake, fake_y_score, mask, 'train', compartment)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    # Gradient penalty calculation outside the main function to avoid variable re-creation
    @tf.function
    def calculate_gradient_penalty(self, x_real, x_fake_masked, lambda_gp):
        alpha = tf.random.uniform([x_real.shape[0], 1, 1, 1, 1], 0., 1.)
        interpolated = alpha * x_real + (1 - alpha) * x_fake_masked

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interpolated_score = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(interpolated_score, [interpolated])[0]
        squared_norm = tf.reduce_sum(tf.square(grads), axis=(2, 3, 4))
        grad_penalty = lambda_gp * tf.reduce_mean((tf.sqrt(squared_norm) - 1.0) ** 2)
        
        return grad_penalty

    #@tf.function
    def train_generator_step_WGAN(self, data_pairs, lambda_gp=10):
        u, v, w, u_hr, v_hr, w_hr, venc, mask, compartment = data_pairs
        x_real = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        with tf.GradientTape() as gen_tape:
            input_data = tf.concat([u, v, w], axis=-1)
            x_fake = self.generator(input_data, training=True)
            x_fake_masked = x_fake * tf.expand_dims(mask, -1) if self.mask_for_disc else x_fake
            real_y_score = self.discriminator(x_real, training=False)
            fake_y_score = self.discriminator(x_fake_masked, training=False)

            w_distance = tf.reduce_mean(real_y_score) - tf.reduce_mean(fake_y_score)

            # Gradient Penalty Calculation
            grad_penalty = self.calculate_gradient_penalty(x_real, x_fake_masked, lambda_gp)
            disc_loss = -w_distance + grad_penalty

            self.loss_metrics[f'train_disc_loss'].update_state(disc_loss)
            self.loss_metrics[f'train_wgan_gradpen'].update_state(grad_penalty)

            # Calculate generator loss
            gen_loss = self.gen_calculate_and_update_metrics_WGAN(x_real, x_fake, fake_y_score, mask, 'train', compartment)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

    #@tf.function
    def train_discriminator_step_WGAN(self, data_pairs, lambda_gp=10):
        u, v, w, u_hr, v_hr, w_hr, venc, mask, compartment = data_pairs
        x_real = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        self.discriminator.trainable = True
        with tf.GradientTape() as disc_tape:
            input_data = tf.concat([u, v, w], axis=-1)

            x_fake = self.generator(input_data, training=True)
            x_fake_masked = x_fake * tf.expand_dims(mask, -1) if self.mask_for_disc else x_fake

            real_y_score = self.discriminator(x_real, training=True)
            fake_y_score = self.discriminator(x_fake_masked, training=True)

            w_distance = tf.reduce_mean(real_y_score) - tf.reduce_mean(fake_y_score)

            # Gradient Penalty Calculation
            grad_penalty = self.calculate_gradient_penalty(x_real, x_fake_masked, lambda_gp)
            disc_loss = -w_distance + grad_penalty

        self.loss_metrics[f'train_disc_loss'].update_state(disc_loss)
        self.loss_metrics[f'train_wgan_gradpen'].update_state(grad_penalty)

        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        self.discriminator.trainable = False

        # Generator - store metrics
        gen_loss = self.gen_calculate_and_update_metrics_WGAN(x_real, x_fake, fake_y_score, mask, 'train', compartment)

    #@tf.function
    def test_step(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs

        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        input_data = tf.concat([u,v,w], axis=-1)
        predictions = self.generator(input_data, training=False)
        
        self.gen_calculate_and_update_metrics(hires, predictions, mask, compartment, 'val')
    
        return predictions
        
    #@tf.function
    def test_step_relativistic(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs
        
        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)

        input_data = tf.concat([u,v,w], axis=-1)

        x_pred = self.generator(input_data, training=False) # x_pred image (Nx24x24x24)
        x_pred_p = x_pred * tf.expand_dims(mask, -1) if self.mask_for_disc else x_pred # masked (Nx24x24x24)
        fake_y_pred = self.discriminator(x_pred_p, training=False)

        real_y_pred = self.discriminator(x_target, training=False)

        disc_loss = self.disc_calculate_and_update_metrics_relativistic(fake_y_pred, real_y_pred, 'val')
        self.loss_metrics['val_disc_loss'].update_state(disc_loss)

        self.gen_calculate_and_update_metrics_relativistic(x_target, x_pred, real_y_pred, fake_y_pred, mask, 'val', compartment)
        return x_pred
    
    #@tf.function
    def test_step_vanilla(self, data_pairs):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs

        x_target = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        input_data = tf.concat([u,v,w], axis=-1)
        x_pred = self.generator(input_data, training=False)
        x_pred_p = x_pred * tf.expand_dims(mask, -1) if self.mask_for_disc else x_pred
        fake_y_pred = self.discriminator(x_pred_p, training=False)

        real_y_target = np.ones(fake_y_pred.shape)
        fake_y_target = np.zeros(fake_y_pred.shape)

        real_y_pred = self.discriminator(x_target, training=False)

        disc_loss = self.disc_calculate_and_update_metrics_vanilla(fake_y_target, fake_y_pred, real_y_target, real_y_pred, 'val')
        self.loss_metrics['val_disc_loss'].update_state(disc_loss)
        
        self.gen_calculate_and_update_metrics_vanilla(x_target, x_pred, real_y_target, fake_y_pred, mask, 'val', compartment)
       
        return x_pred

    @tf.function
    def test_step_WGAN(self, data_pairs, lambda_gp=10):
        u,v,w, u_hr,v_hr, w_hr, venc, mask, compartment = data_pairs
        
        x_real = tf.concat((u_hr, v_hr, w_hr), axis=-1) 
        input_data = tf.concat([u,v,w], axis=-1)
        x_fake = self.generator(input_data, training=False)
        x_fake_masked = x_fake * tf.expand_dims(mask, -1) if self.mask_for_disc else x_fake

        real_y_score = self.discriminator(x_real, training=False)
        fake_y_score = self.discriminator(x_fake_masked, training=False)

        w_distance = tf.reduce_mean(real_y_score) - tf.reduce_mean(fake_y_score)

        # Gradient Penalty Calculation
        grad_penalty = self.calculate_gradient_penalty(x_real, x_fake_masked, lambda_gp)
        disc_loss = -w_distance + grad_penalty

        self.loss_metrics[f'val_disc_loss'].update_state(disc_loss)
        self.loss_metrics[f'val_wgan_gradpen'].update_state(grad_penalty)

        # Generator Training
        self.gen_calculate_and_update_metrics_WGAN(x_real, x_fake, fake_y_score, mask, 'val', compartment)
       
        return x_fake

    def gen_calculate_and_update_metrics(self, hires, predictions, mask, compartment, metric_set):

        loss, mse, fluid_mse, non_fluid_mse, boundary_mse, boundary_rmse, core_mse = self.loss_function(hires, predictions, mask, compartment)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.generator) # * self.genL2weight
            self.loss_metrics[f'l2_gen_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        self.loss_metrics[f'{metric_set}_gen_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse_boundary'].update_state(boundary_mse)
        self.loss_metrics[f'{metric_set}_rmse_boundary'].update_state(boundary_rmse)
        self.loss_metrics[f'{metric_set}_mse_core'].update_state(core_mse)
        self.loss_metrics[f'{metric_set}_mse_fluid'].update_state(fluid_mse)
        self.loss_metrics[f'{metric_set}_mse_nf'].update_state(non_fluid_mse)
        adv_loss = 0
        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)

        return loss

    def gen_calculate_and_update_metrics_relativistic(self, hires, predictions, real_y_pred, fake_y_pred, mask, metric_set, compartment):

        loss, mse, fluid_mse, non_fluid_mse, boundary_mse, boundary_rmse, core_mse = self.loss_function(hires, predictions, mask, compartment)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.generator) # * self.genL2weight
            self.loss_metrics[f'l2_gen_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        # Relativistic adversarial loss:
        mean_fake_y_pred = tf.reduce_mean(fake_y_pred, axis=0, keepdims=True)  # E(D(FAKE))
        mean_real_y_pred = tf.reduce_mean(real_y_pred, axis=0, keepdims=True)  # E(D(REAL))

        # Generator loss for real images (wants D_ra(x_r, x_f) to be classified as fake)
        logits_real = real_y_pred - mean_fake_y_pred
        labels_real = tf.zeros_like(real_y_pred)
        gen_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=labels_real))

        ## Generator loss for fake images (wants D_ra(x_f, x_r) to be classified as real)
        logits_fake = fake_y_pred - mean_real_y_pred
        labels_fake = tf.ones_like(fake_y_pred)
        gen_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=labels_fake))

        ## Combine the losses
        adv_loss = (gen_loss_real + gen_loss_fake) / 2
        adv_loss = self.disc_loss_weight * adv_loss

        # Log and append to total loss
        loss += adv_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        self.loss_metrics[f'{metric_set}_gen_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse_boundary'].update_state(boundary_mse)
        self.loss_metrics[f'{metric_set}_rmse_boundary'].update_state(boundary_rmse)
        self.loss_metrics[f'{metric_set}_mse_core'].update_state(core_mse)
        self.loss_metrics[f'{metric_set}_mse_fluid'].update_state(fluid_mse)
        self.loss_metrics[f'{metric_set}_mse_nf'].update_state(non_fluid_mse)
        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)
        return loss
    
    def gen_calculate_and_update_metrics_vanilla(self, hires, predictions, real_y_target, fake_y_pred, mask, metric_set, compartment):

        loss, mse, fluid_mse, non_fluid_mse, boundary_mse, boundary_rmse, core_mse = self.loss_function(hires, predictions, mask, compartment)
        #loss, mse, divloss = self.loss_function(hires, predictions, mask)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.generator) # * self.genL2weight
            self.loss_metrics[f'l2_gen_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        adv_loss = self.disc_loss_weight * self.calculate_bce(real_y_target, fake_y_pred)
        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)

        loss += adv_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        self.loss_metrics[f'{metric_set}_gen_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse_boundary'].update_state(boundary_mse)
        self.loss_metrics[f'{metric_set}_rmse_boundary'].update_state(boundary_rmse)
        self.loss_metrics[f'{metric_set}_mse_core'].update_state(core_mse)
        self.loss_metrics[f'{metric_set}_mse_fluid'].update_state(fluid_mse)
        self.loss_metrics[f'{metric_set}_mse_nf'].update_state(non_fluid_mse)
        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)
        return loss
    
    def gen_calculate_and_update_metrics_WGAN(self, hires, predictions, fake_y_score, mask, metric_set, compartment):

        loss, mse, fluid_mse, non_fluid_mse, boundary_mse, boundary_rmse, core_mse = self.loss_function(hires, predictions, mask, compartment)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.generator)
            self.loss_metrics[f'l2_gen_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        adv_loss = - self.disc_loss_weight * tf.reduce_mean(fake_y_score)  # Wasserstein Generator Loss

        self.loss_metrics[f'{metric_set}_gen_adv_loss'].update_state(adv_loss)

        loss += adv_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        self.loss_metrics[f'{metric_set}_gen_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse_boundary'].update_state(boundary_mse)
        self.loss_metrics[f'{metric_set}_rmse_boundary'].update_state(boundary_rmse)
        self.loss_metrics[f'{metric_set}_mse_core'].update_state(core_mse)
        self.loss_metrics[f'{metric_set}_mse_fluid'].update_state(fluid_mse)
        self.loss_metrics[f'{metric_set}_mse_nf'].update_state(non_fluid_mse)
        return loss


    def disc_calculate_and_update_metrics_relativistic(self, fake_y_pred, real_y_pred, metric_set):

        mean_fake_y_pred = tf.reduce_mean(fake_y_pred, axis=0, keepdims=True)  # E(D(FAKE))
        mean_real_y_pred = tf.reduce_mean(real_y_pred, axis=0, keepdims=True)  # E(D(REAL))

        # Real relativistic discriminator loss
        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_y_pred - mean_fake_y_pred, labels=tf.ones_like(real_y_pred)))  
        # SCE(logit, label) = -label*log(sigmoid(logit)) - (1-label)log(1 - sigmoid(logit)))
        # -E_xr(log(D_ra(x_r, x_f))) 
        # D_ra (x_r, x_f) = sigmoid ( D (REAL) - E (D(FAKE)) )

        # Fake relativistic discriminator loss 
        disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_y_pred - mean_real_y_pred, labels=tf.zeros_like(fake_y_pred)))
        # -E_xr(log(1 - D_ra(x_f, x_r))) 
        # D_ra (x_f, x_r) = sigmoid ( D (FAKE) - E (D(REAL)) )

        # Average the real and fake losses
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.discriminator) * self.discL2weight
            self.loss_metrics[f'l2_disc_loss'].update_state(l2_reg_loss)
            disc_loss += l2_reg_loss

        self.loss_metrics[f'{metric_set}_disc_loss'].update_state(disc_loss)

        return disc_loss 

    def disc_calculate_and_update_metrics_vanilla(self, fake_y_target, fake_y_pred, real_y_target, real_y_pred, metric_set):
        real_disc_loss = self.calculate_bce(real_y_target, real_y_pred)
        fake_disc_loss = self.calculate_bce(fake_y_target, fake_y_pred)
        adv_loss = 0.5 * (real_disc_loss + fake_disc_loss)
        disc_loss = adv_loss
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss(self.discriminator) * self.discL2weight
            self.loss_metrics[f'l2_disc_loss'].update_state(l2_reg_loss)
            disc_loss += l2_reg_loss

        self.loss_metrics[f'{metric_set}_disc_loss'].update_state(disc_loss)

        return disc_loss

    def reset_metrics(self):
        for key in self.loss_metrics.keys():
            self.loss_metrics[key].reset_states()

    def train_network(self, trainset, valset, cyclic_gen, cyclic_disc, n_epoch,  testset=None):
        """
            Main training function. Receives trainining and validation TF dataset.
        """
        # ----- Run the training -----
        print("==================== TRAINING =================")
        print(f'Learning rate {self.optimizer.lr.numpy():.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()
        
        # Setup acc and data count
        previous_loss = np.inf
        total_batch_train = tf.data.experimental.cardinality(trainset).numpy()
        total_batch_val = tf.data.experimental.cardinality(valset).numpy()

        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            # Reset the metrics at the start of the next epoch
            self.reset_metrics()

            start_loop = time.time()
            # --- Training ---
            for i, (data_pairs) in enumerate(trainset):
                # Train the network
                if epoch < self.epochs_before_disc:
                    self.train_step(data_pairs)
                elif self.adversarial == 'relativistic':
                    self.train_step_relativistic(data_pairs)
                elif self.adversarial == 'WGAN':
                    if cyclic_gen is None or cyclic_disc is None:
                        self.train_step_WGAN(data_pairs)
                    else:
                        # Determine cyclic phase
                        cyclic_phase = epoch % (cyclic_gen + cyclic_disc)
                        train_generator_only = cyclic_phase < cyclic_gen
                        train_discriminator_only = cyclic_gen <= cyclic_phase < (cyclic_gen + cyclic_disc)
                        if train_generator_only:
                            self.train_generator_step_WGAN(data_pairs)
                        elif train_discriminator_only:
                            self.train_discriminator_step_WGAN(data_pairs)
                else:
                    self.train_step_vanilla(data_pairs)


                message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.loss_metrics['train_gen_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Validation ---
            for i, (data_pairs) in enumerate(valset):
                if epoch < self.epochs_before_disc:
                    self.test_step(data_pairs)
                elif self.adversarial == 'relativistic':
                    self.test_step_relativistic(data_pairs)
                elif self.adversarial == 'WGAN':
                    self.test_step_WGAN(data_pairs)
                else:
                    self.test_step_vanilla(data_pairs) 

                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.loss_metrics['val_gen_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Epoch logging ---
            message = f"\rEpoch {epoch+1} Train loss: {self.loss_metrics['train_gen_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %), Val loss: {self.loss_metrics['val_gen_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
            
            loss_values = []
            # Get the loss values from the loss_metrics dict
            for key, value in self.loss_metrics.items():
                # TODO: handle formatting here
                loss_values.append(f'{value.result():.5f}')
            loss_str = ','.join(loss_values)
            log_line = f"{epoch+1},{loss_str},{self.optimizer.lr.numpy():.6f},{time.time()-start_loop:.1f}"
            
            self._update_summary_logging(epoch)

            # --- Save criteria ---
            if self.loss_metrics[self.accuracy_metric].result() < previous_loss:
                self.save_best_model()
                
                # Update best accuracy
                previous_loss = self.loss_metrics[self.accuracy_metric].result()
                
                # Logging
                message  += ' ****' # Mark as saved
                log_line += ',****'

                # Benchmarking
                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy, quick_mse, quick_div = self.quicksave(testset, epoch+1)
                    quick_loss, quick_accuracy, quick_mse, quick_div = np.mean(quick_loss), np.mean(quick_accuracy), np.mean(quick_mse), np.mean(quick_div)

                    message  += f' Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)'
                    log_line += f', {quick_loss:.7f}, {quick_accuracy:.2f}%, {quick_mse:.7f}, {quick_div:.7f}'

            # --- 2nd save criteria ---
            if (epoch+1) % 10 == 0 and epoch != 0:
                self.save_model(epoch+1)

                # Logging
                message  += ' *' # Mark as saved
                log_line += ',*'

            # Logging
            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        # End
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message =  f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += f"\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message)
        
        # Finish!
        
    def save_best_model(self):
        """
            Save model weights and also optmizer weights to enable restore model
            to continue training

            Based on:
            https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        # Save model weights.
        self.model.save(f'{self.model_path}-best.h5')
        
        # Save optimizer weights.
        symbolic_weights = getattr(self.optimizer, 'weights', None)
        if symbolic_weights is None:
            print("WARN --- MANUALFIX USED")
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(f'{self.model_dir}/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

    def save_model(self, epoch_number):
        """
            Save model weights and also optmizer weights to enable restore model
            to continue training

            Based on:
            https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        # Save model weights.
        self.model.save(f'{self.model_path}-epoch' + str(epoch_number) + '.h5')

    def restore_model(self, old_model_dir, old_model_file):
        """
            Restore model weights and optimizer weights for uncompiled model
            Based on: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

            For an uncompiled model, we cannot just set the optmizer weights directly because they are zero.
            We need to at least do an apply_gradients once and then set the optimizer weights.
        """
        # Set the path for the weights and optimizer
        model_weights_path = f"{old_model_dir}/{old_model_file}"

        # Set the trainable weights of the model
        self.model.load_weights(model_weights_path)

    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Filter out the train and val metrics
        train_metrics = {k.replace('train_',''): v for k, v in self.loss_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_',''): v for k, v in self.loss_metrics.items() if k.startswith('val_')}
        
        # Summary writer
        with self.train_writer.as_default():
            for key in train_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  train_metrics[key].result(), step=epoch)         
        
        with self.val_writer.as_default():
            for key in val_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  val_metrics[key].result(), step=epoch)
        
    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark testset.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):
            u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
            hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
            input_data = [u,v,w]

            preds = self.generator.predict(input_data)

            loss_val, mse, divloss = self.loss_function(hires, preds, mask)
            rel_loss = self.accuracy_function(hires, preds, mask)
            # Do only 1 batch
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

        preds = np.expand_dims(preds, 0) # Expand dim to [epoch_nr, batch, ....]
        
        h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds[...,0], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds[...,1], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds[...,2], compression='gzip')

        if epoch_nr == 1:
            # Save the actual data only for the first epoch
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w, compression='gzip')

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", np.squeeze(u_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", np.squeeze(v_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", np.squeeze(w_hr, -1), compression='gzip')
            
            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask, compression='gzip')
        
        return loss_val, rel_loss, mse, divloss