import scipy.misc
import numpy as np
import os
import cv2
import imageio

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(out_path, img)


def get_img(src, img_size=False):
	img = cv2.imread(src)[:,:,(2,1,0)]
	if not (len(img.shape) == 3 and img.shape[2] == 3):
		img = np.dstack((img, img, img))

	if img_size is not False:
		img_target_size = (img_size[0],img_size[1])
		img = cv2.resize(img,img_target_size,interpolation = cv2.INTER_CUBIC)
        
	return img

def resize_img(img_path):
    file_list = list_files(img_path)
    print(file_list)
    for name in file_list:
        img = cv2.imread(img_path+name)
        print(img_path+name)
        print(type(img))
        # cv2.imshow(img)
        # cv2.waitKeys(0)
        
        new_img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        cv2.imwrite('./images/styles_resized/' + name, new_img)

def exists(p, msg):
    assert os.path.exists(p), msg




