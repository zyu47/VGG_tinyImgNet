import tensorflow as tf
import numpy as np
from VGGnet import VGGNet
import single_scale
import os
import random
import cv2


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from tensorflow.python.client import device_lib
#print device_lib.list_local_devices()


data_dir = '/s/red/a/nobackup/vision/jason/tinyImageNet/'
#train_dir = 'tiny-imagenet-200/train/'
#valid_dir = 'tiny-imagenet-200/val/'
logpath = './vgg-net/'

#class_folders = os.listdir(train_dir)
#labels = range(len(class_folders))

def createbatch(xtrain, ytrain, batchsize):
	batch_ind = []
	# Make sure each batch is balanced
	for i in range(200):
		batch_ind.append(np.random.choice(500, 1)[0] + i*500)
	xbatch = []
	for i in batch_ind:
		xbatch.append(single_scale.flip_image(single_scale.crop(single_scale.resize_image(img = xtrain[i], size=75))))
	return np.array(xbatch), ytrain[batch_ind]
    #for ind, folder in enumerate(class_folders):
        #images = os.listdir(os.path.join(train_dir, folder, 'images'))
        #for i in random.sample(images, 1):
            #path = os.path.join(train_dir, folder, 'images', i)
            #processed = single_scale.flip_image(single_scale.crop(single_scale.resize_image(path, size=128)))#(single_scale.crop(single_scale.resize_image(path)))
            #xbatch.append(processed)
            #ybatch.append(ind)
    #sess = tf.Session()
    #return np.array(xbatch), sess.run(tf.one_hot(ybatch, 200))

def valbatch():
    f = open(os.path.join(valid_dir, 'val_annotations.txt'))
    valid_labels_map = {}
    for line in f:
        valid_labels_map[line.split('\t')[0]] =line.split('\t')[1]
    f.close()

    valid_imgs = os.listdir(os.path.join(valid_dir, 'images'))
    xbatch = []
    ybatch = []

    for i in valid_imgs:
        path = os.path.join(valid_dir, 'images', i)
        img_tmp_proc = single_scale.resize_image(path, size=64)
        xbatch.append(img_tmp_proc)

        folder = valid_labels_map[i]
        label = class_folders.index(folder)
        ybatch.append(label)
    sess = tf.Session()
    return np.array(xbatch), sess.run(tf.one_hot(ybatch, 200))

def train_all():
	x = []
	y = []
	for ind, folder in enumerate(class_folders):
		print('Reading Folder %s' %folder)
		images = os.listdir(os.path.join(train_dir, folder, 'images'))
		for i in images:
			print('\tReading Image %s' %i)
			path = os.path.join(train_dir, folder, 'images', i)
			img_tmp_proc = single_scale.resize_image(path, size=64)
			x.append(img_tmp_proc)
			y.append(ind)
	sess = tf.Session()
	np.save('./x.npy', np.array(x))
	np.save('./y.npy', sess.run(tf.one_hot(y, 200)))
	print('training x and y saved!')


def train(vgg):
	#losses = []
	#xvalid, yvalid = valbatch()
	valid_summary = tf.Summary()
	train_act_summary = tf.Summary()
	# validaiton data is mean-centered, but training data is not
	xvalid = np.load(data_dir + 'xvalid_tinyimgnet.npy', encoding='latin1')
	yvalid = np.load(data_dir + 'yvalid_tinyimgnet.npy', encoding='latin1')
	xtrain = np.load(data_dir + 'xtrain_tinyimgnet.npy', encoding='latin1')
	ytrain = np.load(data_dir + 'ytrain_tinyimgnet.npy', encoding='latin1')
	
	mean = np.load('./mean_image_tiny_imagenet.npy')
	xtrain = xtrain - mean
	xtrain_partial_ind = np.random.choice(xtrain.shape[0], 10000)
	
	learning_rate = 1e-4
	#decay = 1e-3
	decay = 0
	batchsize = 200
	
	for step in range(10000):
		xbatch, ybatch = createbatch(xtrain, ytrain, batchsize)
		print(xbatch.shape)
		print(ybatch.shape)
		#print xbatch[0].shape, ybatch[0]
		lr = (learning_rate * 1.0/(1.0 + decay*step))
		print(lr)
		vgg.train(xbatch, ybatch, lr, 0.5, step)

		if step % 100 == 0:
			vgg.test(xvalid, yvalid, step, valid_summary)
			vgg.test(xtrain[xtrain_partial_ind], ytrain[xtrain_partial_ind], step, train_act_summary, train=True)

def train_acc_all(xtrain, vgg):
	x = np.load('./x.npy')
	y = np.load('./y.npy')
	accuracies = []
	losses = []
	for it in range(len(x)/200):
		acc, ls = vgg.sess.run([vgg.accuracy, vgg.loss], {vgg.x:x, vgg.y_:y, vgg.keep_prob:1.0})
		accuracies.append(acc)
		losses.append(ls)
	accuracy = np.mean(accuracies)
	loss = np.mean(losses)
	print('Accuracy: %.4f' %accuracy)
	print('Loss: %.4f' %loss)

def test_v(vgg):
	w11, w21, b11 = vgg.sess.run([vgg.test1, vgg.test2, vgg.test3])
	print(w11)
	print('---------------\n%s' %w21)
	print('---------------\n%s' %b11)
	
	xvalid = np.load(data_dir + 'xvalid_tinyimgnet.npy', encoding='latin1')
	yvalid = np.load(data_dir + 'yvalid_tinyimgnet.npy', encoding='latin1')
	print('validation data loaded')
	valid_summary = tf.Summary()
	vgg.test(xvalid, yvalid, 0, valid_summary)

def test_preproc(iterations = 10):
	xvalid = np.load(data_dir + 'xvalid_tinyimgnet.npy', encoding='latin1')
	print(xvalid.shape)
	for i in range(iterations):
		rand_i = np.random.choice(xvalid.shape[0], 1)[0]
		print(rand_i, xvalid[rand_i].shape)
		cv2.imwrite('./test/ori%d.png' %i, xvalid[rand_i].reshape([64,64,3]))
		#cv2.waitKey(0)
		newimg = single_scale.flip_image(single_scale.crop(single_scale.resize_image(img = xvalid[rand_i], size=75)))
		cv2.imwrite('./test/proc%d.png' %i, newimg)
		#cv2.waitKey(0)
#train_all()
vgg = VGGNet(logpath, test=True, train_conv = True, read_weights = False, from_scratch=False, read_all_weights = False, verbose = False)
#vgg.saver.restore(vgg.sess, logpath)
#train_acc_all(vgg)
train(vgg)
#test_v(vgg)
#test_preproc()
    #if step % 100 == 0:

    # if step < 10:
    #     losses.append(newloss)
    # else:
    #     if np.mean(losses, newloss)


