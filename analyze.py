import tensorflow as tf
import numpy as np

class VGGNet:
	def __init__(self, logpath):
		train_conv = False
		self.logs_path = logpath
		with tf.device('/gpu:0'):
			self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
			self.y_ = tf.placeholder(tf.float32, [None, 200])
			self.keep_prob = tf.placeholder(tf.float32)
			self.reg = 5e-4

			#Convolutional layer no 1
			self.W1_1 = tf.get_variable("W1_1", shape=[3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b1_1 = tf.Variable(tf.constant(0.01, shape=[64]), trainable = train_conv, name = 'b1_1')
			self.conv3_64 = tf.nn.conv2d(self.x, W1_1, strides = [1,1,1,1], padding='SAME')
			self.conv3_64_relu = tf.nn.relu(conv3_64 + b1_1)
			
			self.parameters += [W1_1, b1_1]

			# Convolutional layer no 2
			self.W1_2 = tf.get_variable("W1_2", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b1_2 = tf.Variable(tf.constant(0.01, shape=[64]), trainable = train_conv, name = 'b1_2')
			self.conv3_64_2 = tf.nn.conv2d(conv3_64_relu, W1_2, strides=[1, 1, 1, 1], padding='SAME')
			self.conv3_64_2_relu = tf.nn.relu(conv3_64_2 + b1_2)
			
			self.parameters += [W1_2, b1_2]

			#Max pooling layer
			self.max_pool_2x2_1 = tf.nn.max_pool(conv3_64_2_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions:  32x32

			#convolutional layer no 3
			self.W2_1 = tf.get_variable("W2_1", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b2_1 = tf.Variable(tf.constant(0.01, shape=[128]), trainable = train_conv, name='b2_1')
			self.conv3_128 = tf.nn.conv2d(max_pool_2x2_1, W2_1, strides = [1,1,1,1], padding='SAME')
			self.conv3_128_relu = tf.nn.relu(conv3_128 + b2_1)
			
			self.parameters += [W2_1, b2_1]

			# convolutional layer no 4
			self.W2_2 = tf.get_variable("W2_2", shape=[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b2_2 = tf.Variable(tf.constant(0.01, shape=[128]), trainable = train_conv, name='b2_2')
			self.conv3_128_2 = tf.nn.conv2d(conv3_128_relu, W2_2, strides=[1, 1, 1, 1], padding='SAME')
			self.conv3_128_2_relu = tf.nn.relu(conv3_128_2 + b2_2)
			
			self.parameters += [W2_2, b2_2]

			#Max pooling layer
			self.max_pool_2x2_2 = tf.nn.max_pool(conv3_128_2_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 16x16


			# convolutional layer No5
			self.W3_1 = tf.get_variable("W3_1", shape=[3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b3_1 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name='b3_1')
			self.conv3_256 = tf.nn.conv2d(max_pool_2x2_2, W3_1, strides = [1,1,1,1], padding='SAME')
			self.conv3_256_relu = tf.nn.relu(conv3_256 + b3_1)
			
			self.parameters += [W3_1, b3_1]

			# convolutional layer No6
			self.W3_2 = tf.get_variable("W3_2", shape=[3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b3_2 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name= 'b3_2')
			self.conv3_256_2 = tf.nn.conv2d(conv3_256_relu, W3_2, strides=[1, 1, 1, 1], padding='SAME')
			self.conv3_256_2_relu = tf.nn.relu(conv3_256_2 + b3_2)
			
			self.parameters += [W3_2, b3_2]

			#convolutional layer No7
			self.W3_3 = tf.get_variable("W3_3", shape=[3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b3_3 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name='b3_3')
			self.conv3_256_3 = tf.nn.conv2d(conv3_256_2_relu, W3_3, strides = [1,1,1,1], padding='SAME')
			self.conv3_256_3_relu = tf.nn.relu(conv3_256_3 + b3_3)
			
			self.parameters += [W3_3, b3_3]

			#Max pooling layer
			self.max_pool_2x2_3 = tf.nn.max_pool(conv3_256_3_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 8x8


			# convolutional layer No8
			self.W4_1 = tf.get_variable("W4_1", shape=[3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b4_1 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv)
			self.conv3_512 = tf.nn.conv2d(max_pool_2x2_3, W4_1, strides = [1,1,1,1], padding='SAME')
			self.conv3_512_relu = tf.nn.relu(conv3_512 + b4_1)
			
			self.parameters += [W4_1, b4_1]

			# convolutional layer No9
			self.W4_2 = tf.get_variable("W4_2", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b4_2 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv, name='b4_2')
			self.conv3_512_2 = tf.nn.conv2d(conv3_512_relu, W4_2, strides=[1, 1, 1, 1], padding='SAME')
			self.conv3_512_2_relu = tf.nn.relu(conv3_512_2 + b4_2)
			
			self.parameters += [W4_2, b4_2]

			#convolutional layer No10
			self.W4_3 = tf.get_variable("W4_3", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b4_3 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv, name='b4_3')
			self.conv3_512_3 = tf.nn.conv2d(conv3_512_2_relu, W4_3, strides = [1,1,1,1], padding='SAME')
			self.conv3_512_3_relu = tf.nn.relu(conv3_512_3 + b4_3)
			
			self.parameters += [W4_3, b4_3]

			#Max pooling layer
			self.max_pool_2x2_4 = tf.nn.max_pool(conv3_512_3_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 4x4


			
			# convolutional layer No11
			self.W5_1 = tf.get_variable("W5_1", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b5_1 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv, name='b5_1')
			self.conv3_512_4 = tf.nn.conv2d(max_pool_2x2_4, W5_1, strides = [1,1,1,1], padding='SAME')
			self.conv3_512_4_relu = tf.nn.relu(conv3_512_4 + b5_1)
			
			self.parameters += [W5_1, b5_1]

			# convolutional layer No12
			self.W5_2 = tf.get_variable("W5_2", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b5_2 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv,name='b5_2')
			self.conv3_512_5 = tf.nn.conv2d(conv3_512_4_relu, W5_2, strides=[1, 1, 1, 1], padding='SAME')
			self.conv3_512_5_relu = tf.nn.relu(conv3_512_5 + b5_2)
			
			self.parameters += [W5_2, b5_2]

			#convolutional layer No13
			self.W5_3 = tf.get_variable("W5_3", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			self.b5_3 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv, name='b5_3')
			self.conv3_512_6 = tf.nn.conv2d(conv3_512_5_relu, W5_3, strides = [1,1,1,1], padding='SAME')
			self.conv3_512_6_relu = tf.nn.relu(conv3_512_6 + b5_3)
			
			self.parameters += [W5_3, b5_3]

			#Max pooling layer
			max_pool_2x2_5 = tf.nn.max_pool(conv3_512_6_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
			if test:
				train_conv = False
			#current image dimensions: 2x2
			if not train_conv:
				tmp = set(tf.global_variables())
			
			#Fully connected layer 1
			W_fc1 = tf.get_variable("W_fc1", shape=[2*2*512, 4096], initializer=tf.contrib.layers.xavier_initializer())
			b_fc1 = tf.Variable(tf.constant(0.01, shape=[4096]), name='b_fc1')

			h_pool5_flat = tf.reshape(max_pool_2x2_5, [-1, 2*2*512])
			h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool5_flat, W_fc1, b_fc1))

			h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

			#Fully connected layer 2
			W_fc2 = tf.get_variable("W_fc2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
			b_fc2 = tf.Variable(tf.constant(0.01, shape=[4096]), name='b_fc2')

			h_fc2 = tf.nn.relu(tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2))
			h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)


			#Fully connected layer 3
			W_fc3 = tf.get_variable("W_fc3", shape=[4096, 200], initializer=tf.contrib.layers.xavier_initializer())
			b_fc3 = tf.Variable(tf.constant(0.01, shape=[200]), name='b_fc3')

			h_fc3 = tf.nn.xw_plus_b(h_fc2_drop, W_fc3, b_fc3)
			
			#Initialize with existing model
			if train_conv:				
				tmp = set(tf.global_variables())
				self.saver_restore = tf.train.Saver(tmp)

			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y_, logits=h_fc3))

			self.loss = cross_entropy

			if not train_conv:			
				self.loss += (l2(self.reg, W_fc3) + l2(self.reg, W_fc2) + l2(self.reg, W_fc1))
			else:
				self.loss += (l2(self.reg, W_fc3) + l2(self.reg, W_fc2) + l2(self.reg, W_fc1) +
							  l2(self.reg, W5_3)  + l2(self.reg, W5_2)  + l2(self.reg, W5_1)   +
							  l2(self.reg, W4_3)  + l2(self.reg, W4_2)  + l2(self.reg, W4_1)   +
							  l2(self.reg, W3_3)  + l2(self.reg, W3_2)  + l2(self.reg, W3_1)   +
							  l2(self.reg, W2_2)  + l2(self.reg, W2_1)  + 
							  l2(self.reg, W1_2)  + l2(self.reg, W1_1))

		correct_prediction = tf.cast(tf.equal(tf.argmax(h_fc3, 1), tf.argmax(self.y_, 1)), tf.float32)
		self.accuracy = tf.reduce_mean(correct_prediction)

		try:
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config= config)
		except:
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config= config)

		# model saver
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, self.logs_path + 'model.ckpt')
		
		# initialization
		#print('----------Initializing weights--------')
		#if from_scratch:
			#self.sess.run(tf.global_variables_initializer())
			#if verbose:
				#print('Train From Scratch')
		#elif not train_conv:
			#if verbose:
				#print('Not Train Convolutional layer')
			#if not read_weights:
				#weights = np.load(self.weight_file)
				#keys = sorted(weights.keys())
				#for i, k in enumerate(keys):
					#if i == len(self.parameters):
						#break
					#print('%d%s%s' %(i,k,np.shape(weights[k])))
					#self.sess.run(self.parameters[i].assign(weights[k]))
				#self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tmp))
			#else:
				#self.saver.restore(self.sess, self.logs_path + 'model.ckpt')
		#else:
			#if verbose:
				#print('Train Every layer')
			#self.saver_restore.restore(self.sess, self.logs_path + 'model.ckpt')
			#self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tmp))

		#if read_all_weights:
			#self.saver.restore(self.sess, self.logs_path + 'model.ckpt')

		print('----------weights initialized---------')
