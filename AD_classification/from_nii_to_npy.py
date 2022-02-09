""""""
import os
import numpy as np
import nibabel as nib
from glob import glob as gg
import pandas as pd
import sys

data_path = str(sys.argv[1])

saver_path = data_path + 'dataset/'
if not os.path.exists(saver_path):
    os.makedirs(saver_path)

files = gg(data_path + '*Scaled_Br*.nii')

img = [nib.load(f) for f in files]

data = [np.array(i.dataobj).astype('float32') for i in img]

label_dic = {'CN': 0, 'MCI': 1, 'AD': 2}

mri_table = pd.read_csv(data_path + 'ADNI1_Screening_1.5T_4_22_2021.csv')
subjects = mri_table['Subject']
subject_groups = mri_table['Group']

j = 0
for name in files:
    subject_name = os.path.basename(name).split('ADNI_')[1].split('_MR')[0]
    correspondence = np.where(np.array(subjects) == subject_name)[0]
    label = label_dic[subject_groups[correspondence[0]]]
    saver_name = str(label) + '_' + name.split('/')[-1].split('.nii')[0]
    np.save(saver_path + saver_name, (data[j], label))
    j += 1

# show the size of the images
all_possible_sizes = list(set(list([d.shape for d in data])))
n_sizes = len(all_possible_sizes)

sizes = [[] for i in range(n_sizes)]
for i in range(len(data)):
    s = data[i].shape
    for n in range(n_sizes):
        if s == all_possible_sizes[n]:
            sizes[n].append(i)

dim = [len(i) for i in sizes]

textfile = open(data_path + 'dataset_info.txt', 'w')

for i in range(n_sizes):
    print(str(dim[i]) + ' images have size ' + str(all_possible_sizes[i]))
    textfile.write(str(dim[i]) + ' images have size ' + str(all_possible_sizes[i]) + '\n')

textfile.write('\n')
for i in range(n_sizes):
    textfile.write('List of subjects with MRI size equal to ' + str(all_possible_sizes[i]) + ': \n\n')
    for j in sizes[i]:
        textfile.write(files[j].split('/')[-1].split('.nii')[0] + '\n')
    textfile.write('\n')

''' 
Output:

81 images have size (256, 256, 170)
2 images have size (256, 256, 160)
271 images have size (256, 256, 166)
1 images have size (256, 256, 161)
1 images have size (256, 256, 162)
320 images have size (192, 192, 160)
1 images have size (256, 256, 146)
12 images have size (256, 256, 184)
119 images have size (256, 256, 180)

'''
