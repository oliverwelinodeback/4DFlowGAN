import numpy as np
import h5py
import PatchData as pd
import pandas

def load_data(input_filepath):
    with h5py.File(input_filepath, mode = 'r' ) as hdf5:
        data_nr = len(hdf5['u'])

    indexes = np.arange(data_nr)
    print("Dataset: {} rows".format(len(indexes)))
    return indexes

if __name__ == "__main__": 
    patch_size = 12 # Patch size, this will be checked to make sure the generated patches do not go out of bounds
    n_patch = 10 # number of patch per time frame
    n_empty_patch_allowed = 1 # max number of empty patch per frame
    all_rotation = True # When true, include 90,180, and 270 rotation for each patch. When False, only include 1 random rotation.
    mask_threshold = 0.5 # Threshold for non-binary mask 
    minimum_coverage = 0.05 # Minimum fluid region within a patch. Any patch with less than this coverage will not be taken. Range 0-1

    base_path = '../../data' 

    #LowRes velocity data
    lr_files = [
                'example_data_LR.h5',
                #'patient3-postOp_LR_dv.h5',
                ] 

    #HiRes velocity data
    hr_files = [
                'example_data_HR.h5',
                #"patient3-postOp_HR.h5",
                ] 
    
    output_filename = f'{base_path}/validate.csv'

    # Prepare the CSV output
    pd.write_header(output_filename)

    for lr_file, hr_file in zip(lr_files, hr_files):

        # Load the data
        input_filepath = f'{base_path}/{lr_file}'
        file_indexes = load_data(input_filepath)

        # because the data is homogenous in 1 table, we only need the first data
        with h5py.File(input_filepath, mode = 'r' ) as hdf5:
            print([key for key in hdf5.keys()])
            mask = np.asarray(hdf5['mask'])
            if len(mask.shape) == 4: 
                mask = mask[0]
        # We basically need the mask on the lowres data, the patches index are retrieved based on the LR data.
        print("Overall shape", mask.shape)

        # Do the thresholding
        binary_mask = (mask >= mask_threshold) * 1

        # Generate random patches for all time frames
        for index in file_indexes:
            print('Generating patches for row', index)
            pd.generate_random_patches(lr_file, hr_file, output_filename, index, n_patch, binary_mask, 
                                    patch_size, minimum_coverage, n_empty_patch_allowed, all_rotation)
            
        print(f'Done processing {lr_file} and {hr_file}. Saved in {output_filename}')

    # Save settings to csv
    settings = {
        'patch_size': patch_size,
        'n_patch': n_patch,
        'n_empty_patch_allowed': n_empty_patch_allowed,
        'all_rotation': all_rotation,
        'mask_threshold': mask_threshold,
        'minimum_coverage': minimum_coverage
    }

    settings_df = pandas.DataFrame(list(settings.items()), columns=['Metric', 'Value'])

    # Combine the file lists with the settings
    # for lr, hr in zip(lr_files, hr_files):
    #     row = {'lr_file': lr, 'hr_file': hr}
    #     settings_df = pandas.concat([settings_df, pandas.DataFrame([row])], ignore_index=True)

    settings_filename = output_filename.replace('.csv', '_settings.csv')
    settings_df.to_csv(settings_filename, index=False)


