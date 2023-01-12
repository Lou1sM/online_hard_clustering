from PIL import Image
from pdb import set_trace
import json
import numpy as np
from os import listdir
from os.path import join
from dl_utils.misc import check_dir


jpg_dir = 'tiny-imagenet-200/train'
with open('tiny-imagenet-200/words.txt') as f:d=f.readlines()
d1=[x[:-1].split('\t') for x in d]
wd={k:v for (k,v) in d1}
class_wn_synsets = [wd[x] for x in sorted(list(set(listdir(jpg_dir))))]
with open('tiny-imagenet-200/synsets_dict.json','w') as f: json.dump(class_wn_synsets,f)

check_dir('tiny-imagenet-200/np_data/')
for wn_name in listdir(jpg_dir):
    class_data_as_list = []
    for fname in listdir(join(jpg_dir,wn_name,'images')):
        img = Image.open((join(jpg_dir,wn_name,'images',fname)))
        np_img = np.asarray(img)
        class_name = wd[wn_name]
        class_idx = class_wn_synsets.index(class_name)
        if np_img.ndim==2:
            np_img = np.repeat(np.expand_dims(np_img,2),3,axis=2)
        assert np_img.shape == (64,64,3)
        class_data_as_list.append(np_img)
    class_data_as_array = np.stack(class_data_as_list)
    class_labels_as_array = np.tile(np.array([class_idx]),len(class_data_as_array))
    np.save(f'tiny-imagenet-200/np_data/{class_idx}.npy', class_data_as_array)
    np.save(f'tiny-imagenet-200/np_data/{class_idx}_labels.npy', class_labels_as_array)
