import numpy as np
from glob import glob
import PIL.Image as Image
import os
from tqdm import tqdm

save_dir = 'd503/test'
save_flow_dir = 'd503_flow/test'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_flow_dir, exist_ok=True)

dataset = glob('datasets/d503_SRFormer_format/train/*.png')
dataset.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

for file in tqdm(dataset):
    image = Image.open(file)
    image = image.convert('RGB')
    if image.size[0] * image.size[1] > 1920 * 1080:
        image = image.resize((1920, 1080))
    shape = image.size[1], image.size[0]
    write_dir = f"{save_dir}/{file.split('/')[-1].split('.')[0]}/sharp"
    os.makedirs(write_dir, exist_ok=True)
    image.save(f"{write_dir}/{file.split('/')[-1]}")
    
    direction = np.random.uniform(0, 2*np.pi)
    magnitude = np.random.uniform(0, 0.1) * 147 # normalization constant

    condition = np.zeros((3,) + shape, dtype=np.float32)
    dir_noise = np.random.uniform(0.9, 1.1, size=shape)
    magnitude_noise = np.random.uniform(0.9, 1.1, size=shape)
    condition[0] = np.cos(dir_noise * direction)
    condition[1] = np.sin(dir_noise * direction)
    condition[2] = magnitude_noise * magnitude

    write_dir = f"{save_flow_dir}/{file.split('/')[-1].split('.')[0]}"
    os.makedirs(write_dir, exist_ok=True)
    np.save(f"{write_dir}/{file.split('/')[-1].split('.')[0]}.npy", condition)
