import tensorflow as tf
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import random

root = '../tiny-imagenet-200/train/'

def resize_image(img = None, path = None, size=256, verbose = False, sub_mean = False):
    if path != None:
        img = cv2.imread(path)
    if verbose:
        print(img[0,:,0])
        #for i in range(3):
        print(mean[0,:, 0])
    if sub_mean:
        mean = np.load('./mean_image_tiny_imagenet.npy')
        img = img - mean
    if verbose:
        print(img[0,:,0])
    height, width, channels = img.shape
    if verbose:
        print(img.shape)
    shorter_side = min(height, width)
    if shorter_side == height:
        new_height = size
        new_width = (width*1.0/height)*size
    else:
        new_width = size
        new_height = (height*1.0/width) * size
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    #new_img = tf.image.resize_images(img, size=[int(new_height),int(new_width)], method=tf.image.ResizeMethod.BICUBIC)
    #new_img = tf.image.resize_images(img, size=[65,65], method=tf.image.ResizeMethod.BICUBIC)
    new_img = cv2.resize(img, (int(new_width),int(new_height)), interpolation=cv2.INTER_CUBIC)
    return new_img

def crop(img, size = 64, verbose = False):
    h, w, c = img.shape
    topleft_h = random.randint(0,h-size-1)
    topleft_w = random.randint(0, w-size-1)
    cropped = img[topleft_h:topleft_h + size, topleft_w:topleft_w + size, :]  # tf.image.crop_to_bounding_box(img, topleft_h, topleft_w, 224, 224)
    return cropped

def flip_image(img, verbose=False):
    flag = random.randint(0,1)
    if flag:
        img_flipped = cv2.flip(img, 1)
        return img_flipped
    else:
        return img



def maintest():
    sess = tf.InteractiveSession()
    test = resize_image('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/tiny-imagenet-200/train/n01443537/images/n01443537_4.JPEG', verbose=True)
    for i in range(5):
        test_tf = flip_image(crop(test))
        test_ = sess.run(test_tf)
        cv2.imshow('test image', test_)
        cv2.waitKey(0)
        #print type(test_)
        # plt.imshow(test_)
        # plt.show()


#maintest()
