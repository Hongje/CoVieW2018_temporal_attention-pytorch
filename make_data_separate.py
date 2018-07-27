import numpy as np
import os


data_path = 'data/train'
make_data_path = 'data/train_separate'

data_files = os.listdir(data_path)
data_files.sort()

print(data_files)
print(data_files[0])

for i, file_name in enumerate(data_files):
    if i >= 7:
        full_filename = os.path.join(data_path, file_name)
        print(full_filename)
        npz_file = np.load(full_filename)
        # print(npz_file.files)
        for npz_idx in range(len(npz_file['ids'])):
            if npz_idx % 100 == 0:
                print(file_name, npz_idx)

            np.savez(os.path.join(make_data_path, 'train'+npz_file['ids'][npz_idx]+'.npz'),
                    labels=npz_file['labels'][npz_idx],
                    rgb=npz_file['rgb'][npz_idx],
                    audio=npz_file['audio'][npz_idx])
