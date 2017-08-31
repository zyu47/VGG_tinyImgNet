import tensorflow as tf
import numpy as np

def l1(reg, tensor):
	return reg * tf.reduce_sum(tf.abs(tensor))
def l2(reg, tensor):
	return reg * tf.nn.l2_loss(tensor)
	
class VGGNet:
	def __init__(self, logpath, test = False, train_conv = False, read_weights = False, from_scratch=False, read_all_weights = False, verbose = False):
		if test:
			train_conv = True
		self.logs_path = logpath
		self.weight_file = '/s/red/a/nobackup/vision/jason/vgg16_weights.npz'
		self.parameters = []
		#train_conv = True
		with tf.device('/gpu:0'):
			self.learning_rate = tf.placeholder(tf.float32, [])

		with tf.device('/gpu:0'):
			self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
			self.y_ = tf.placeholder(tf.float32, [None, 200])
			self.keep_prob = tf.placeholder(tf.float32)
			self.reg = 5e-4

			#Convolutional layer no 1
			W1_1 = tf.get_variable("W1_1", shape=[3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.1))
			b1_1 = tf.Variable(tf.constant(0.01, shape=[64]), trainable = train_conv, name = 'b1_1')
			conv3_64 = tf.nn.conv2d(self.x, W1_1, strides = [1,1,1,1], padding='SAME')
			conv3_64_relu = tf.nn.relu(conv3_64 + b1_1)
			
			self.parameters += [W1_1, b1_1]

			# Convolutional layer no 2
			W1_2 = tf.get_variable("W1_2", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
			b1_2 = tf.Variable(tf.constant(0.01, shape=[64]), trainable = train_conv, name = 'b1_2')
			conv3_64_2 = tf.nn.conv2d(conv3_64_relu, W1_2, strides=[1, 1, 1, 1], padding='SAME')
			conv3_64_2_relu = tf.nn.relu(conv3_64_2 + b1_2)
			
			self.parameters += [W1_2, b1_2]

			#Max pooling layer
			max_pool_2x2_1 = tf.nn.max_pool(conv3_64_2_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions:  32x32

			#convolutional layer no 3
			W2_1 = tf.get_variable("W2_1", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
			b2_1 = tf.Variable(tf.constant(0.01, shape=[128]), trainable = train_conv, name='b2_1')
			conv3_128 = tf.nn.conv2d(max_pool_2x2_1, W2_1, strides = [1,1,1,1], padding='SAME')
			conv3_128_relu = tf.nn.relu(conv3_128 + b2_1)
			
			self.parameters += [W2_1, b2_1]

			# convolutional layer no 4
			W2_2 = tf.get_variable("W2_2", shape=[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
			b2_2 = tf.Variable(tf.constant(0.01, shape=[128]), trainable = train_conv, name='b2_2')
			conv3_128_2 = tf.nn.conv2d(conv3_128_relu, W2_2, strides=[1, 1, 1, 1], padding='SAME')
			conv3_128_2_relu = tf.nn.relu(conv3_128_2 + b2_2)
			
			self.parameters += [W2_2, b2_2]

			#Max pooling layer
			max_pool_2x2_2 = tf.nn.max_pool(conv3_128_2_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 16x16


			# convolutional layer No5
			W3_1 = tf.get_variable("W3_1", shape=[3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3, 128, 256], stddev=0.1))
			b3_1 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name='b3_1')
			conv3_256 = tf.nn.conv2d(max_pool_2x2_2, W3_1, strides = [1,1,1,1], padding='SAME')
			conv3_256_relu = tf.nn.relu(conv3_256 + b3_1)
			
			self.parameters += [W3_1, b3_1]

			# convolutional layer No6
			W3_2 = tf.get_variable("W3_2", shape=[3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
			b3_2 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name= 'b3_2')
			conv3_256_2 = tf.nn.conv2d(conv3_256_relu, W3_2, strides=[1, 1, 1, 1], padding='SAME')
			conv3_256_2_relu = tf.nn.relu(conv3_256_2 + b3_2)
			
			self.parameters += [W3_2, b3_2]

			#convolutional layer No7
			W3_3 = tf.get_variable("W3_3", shape=[3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3,256, 256], stddev=0.1))
			b3_3 = tf.Variable(tf.constant(0.01, shape=[256]), trainable = train_conv, name='b3_3')
			conv3_256_3 = tf.nn.conv2d(conv3_256_2_relu, W3_3, strides = [1,1,1,1], padding='SAME')
			conv3_256_3_relu = tf.nn.relu(conv3_256_3 + b3_3)
			
			self.parameters += [W3_3, b3_3]

			#Max pooling layer
			max_pool_2x2_3 = tf.nn.max_pool(conv3_256_3_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 8x8


			# convolutional layer No8
			W4_1 = tf.get_variable("W4_1", shape=[3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3, 256, 512], stddev=0.1))
			b4_1 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv)
			conv3_512 = tf.nn.conv2d(max_pool_2x2_3, W4_1, strides = [1,1,1,1], padding='SAME')
			conv3_512_relu = tf.nn.relu(conv3_512 + b4_1)
			
			self.parameters += [W4_1, b4_1]

			# convolutional layer No9
			W4_2 = tf.get_variable("W4_2", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
			b4_2 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv, name='b4_2')
			conv3_512_2 = tf.nn.conv2d(conv3_512_relu, W4_2, strides=[1, 1, 1, 1], padding='SAME')
			conv3_512_2_relu = tf.nn.relu(conv3_512_2 + b4_2)
			
			self.parameters += [W4_2, b4_2]

			#convolutional layer No10
			W4_3 = tf.get_variable("W4_3", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)#tf.Variable(tf.truncated_normal([3,3,512, 512], stddev=0.1))
			b4_3 = tf.Variable(tf.constant(0.01, shape=[512]), trainable = train_conv, name='b4_3')
			conv3_512_3 = tf.nn.conv2d(conv3_512_2_relu, W4_3, strides = [1,1,1,1], padding='SAME')
			conv3_512_3_relu = tf.nn.relu(conv3_512_3 + b4_3)
			
			self.parameters += [W4_3, b4_3]

			#Max pooling layer
			max_pool_2x2_4 = tf.nn.max_pool(conv3_512_3_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')

			#current image dimensions: 4x4


			
			# convolutional layer No11
			W5_1 = tf.get_variable("W5_1", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			b5_1 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv, name='b5_1')
			conv3_512_4 = tf.nn.conv2d(max_pool_2x2_4, W5_1, strides = [1,1,1,1], padding='SAME')
			conv3_512_4_relu = tf.nn.relu(conv3_512_4 + b5_1)
			
			self.parameters += [W5_1, b5_1]

			# convolutional layer No12
			W5_2 = tf.get_variable("W5_2", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			b5_2 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv,name='b5_2')
			conv3_512_5 = tf.nn.conv2d(conv3_512_4_relu, W5_2, strides=[1, 1, 1, 1], padding='SAME')
			conv3_512_5_relu = tf.nn.relu(conv3_512_5 + b5_2)
			
			self.parameters += [W5_2, b5_2]

			#convolutional layer No13
			W5_3 = tf.get_variable("W5_3", shape=[3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable = train_conv)
			b5_3 = tf.Variable(tf.constant(0.1, shape=[512]), trainable = train_conv, name='b5_3')
			conv3_512_6 = tf.nn.conv2d(conv3_512_5_relu, W5_3, strides = [1,1,1,1], padding='SAME')
			conv3_512_6_relu = tf.nn.relu(conv3_512_6 + b5_3)
			
			self.parameters += [W5_3, b5_3]

			#Max pooling layer
			max_pool_2x2_5 = tf.nn.max_pool(conv3_512_6_relu, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
			if test:
				train_conv = False
			#current image dimensions: 2x2
			if not train_conv:
				tmp = set(tf.global_variables())
			
			#Fully connected layer 1
			W_fc1 = tf.get_variable("W_fc1", shape=[2*2*512, 4096], initializer=tf.contrib.layers.xavier_initializer())#tf.Variable(tf.truncated_normal([4*4*512, 4096], stddev=0.1))
			b_fc1 = tf.Variable(tf.constant(0.01, shape=[4096]), name='b_fc1')

			h_pool5_flat = tf.reshape(max_pool_2x2_5, [-1, 2*2*512])
			h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool5_flat, W_fc1, b_fc1))

			h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

			#Fully connected layer 2
			W_fc2 = tf.get_variable("W_fc2", shape=[4096, 4096], initializer=tf.contrib.layers.xavier_initializer())#tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1))
			b_fc2 = tf.Variable(tf.constant(0.01, shape=[4096]), name='b_fc2')

			h_fc2 = tf.nn.relu(tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2))
			h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)


			#Fully connected layer 3
			W_fc3 = tf.get_variable("W_fc3", shape=[4096, 200], initializer=tf.contrib.layers.xavier_initializer())#tf.Variable(tf.truncated_normal([4096, 200], stddev=0.1))
			b_fc3 = tf.Variable(tf.constant(0.01, shape=[200]), name='b_fc3')

			h_fc3 = tf.nn.xw_plus_b(h_fc2_drop, W_fc3, b_fc3)
			
			#Initialize with existing model
			if train_conv:				
				tmp = set(tf.global_variables())
				self.saver_restore = tf.train.Saver(tmp)

			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y_, logits=h_fc3))
			#cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(h_fc3), reduction_indices=[1]))

			self.loss = cross_entropy

			#L1 and L2 regularization:

			#self.loss += (l1(1e-7, W5_1) + l2(1e-7, W5_1) + 
						 #l1(1e-6, W5_2) + l2(1e-6, W5_2) + 
						 #l1(1e-5, W5_3) + l2(1e-5, W5_3) + 
						 #l1(1e-4, W_fc1) + l2(1e-5, W_fc1) + 
						 #l1(1e-4, W_fc2) + l2(1e-4, W_fc2))
			#L2 self.regularization loss for all weights
			if not train_conv:			
				self.loss += (l2(self.reg, W_fc3) + l2(self.reg, W_fc2) + l2(self.reg, W_fc1))
			else:
				self.loss += (l2(self.reg, W_fc3) + l2(self.reg, W_fc2) + l2(self.reg, W_fc1) +
							  l2(self.reg, W5_3)  + l2(self.reg, W5_2)  + l2(self.reg, W5_1)   +
							  l2(self.reg, W4_3)  + l2(self.reg, W4_2)  + l2(self.reg, W4_1)   +
							  l2(self.reg, W3_3)  + l2(self.reg, W3_2)  + l2(self.reg, W3_1)   +
							  l2(self.reg, W2_2)  + l2(self.reg, W2_1)  + 
							  l2(self.reg, W1_2)  + l2(self.reg, W1_1))

		#with tf.device('/gpu:0'):
		self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov = True).minimize(self.loss)
		#self.train_step = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		#self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

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
		# session
		#self.sess = tf.InteractiveSession()

		# summaries
		tf.summary.scalar('accuracy', self.accuracy)
		tf.summary.scalar('loss', self.loss)
		self.summaries = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.logs_path + '/train', self.sess.graph)
		self.test_writer = tf.summary.FileWriter(self.logs_path + '/test', self.sess.graph)
		self.train_actual_writer = tf.summary.FileWriter(self.logs_path + './train_actual', self.sess.graph)

		# model saver
		self.saver = tf.train.Saver()

		# initialization
		print('----------Initializing weights--------')
		if from_scratch:
			self.sess.run(tf.global_variables_initializer())
			if verbose:
				print('Train From Scratch')
		elif not train_conv:
			if verbose:
				print('Not Train Convolutional layer')
			if not read_weights:
				weights = np.load(self.weight_file)
				keys = sorted(weights.keys())
				for i, k in enumerate(keys):
					if i == len(self.parameters):
						break
					print('%d%s%s' %(i,k,np.shape(weights[k])))
					self.sess.run(self.parameters[i].assign(weights[k]))
				self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tmp))
			else:
				self.saver.restore(self.sess, self.logs_path + 'model.ckpt')
		else:
			if verbose:
				print('Train Every layer')
			self.saver_restore.restore(self.sess, self.logs_path + 'model.ckpt')
			self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tmp))

		if read_all_weights:
			self.saver.restore(self.sess, self.logs_path + 'model.ckpt')

		print('----------weights initialized---------')
		# test
		self.test1 = W1_1
		self.test2 = W2_1
		self.test3 = b1_1
		

		
	def train(self, xbatch, ybatch, learning_rate, keep_prob, step):
		_, summ, acc, ls = self.sess.run([self.train_step, self.summaries, self.accuracy, self.loss],
					  feed_dict = {self.x:xbatch, self.y_:ybatch, self.learning_rate:learning_rate, self.keep_prob:keep_prob})
		self.train_writer.add_summary(summ, step)
		print('Training step %s: \t accuracy:%.4f \t loss:%.4f' %(step, acc, ls))

		if step % 500 == 0:
			model_dir = self.saver.save(self.sess, self.logs_path + 'model.ckpt')
			print('Model saved at %s' %model_dir)
		return ls

	def test(self, xbatch, ybatch, step, summary, train = False):
		accuracies = []
		losses = []

		for it in range(50):
			x_valid_data, yvalid_lbl = xbatch[(it * 200): (it + 1) * 200, :, :, :], ybatch[(it * 200): (it + 1) * 200]
			summ, acc, ls = self.sess.run([self.summaries, self.accuracy, self.loss],
					  feed_dict = {self.x:x_valid_data, self.y_:yvalid_lbl, self.keep_prob:1.0})
			accuracies.append(acc)
			losses.append(ls)

		# Take the mean of you measure
		accuracy = np.mean(accuracies)
		loss = np.mean(losses)

		# Create a new Summary object with your measure

		summary.value.add(tag="accuracy", simple_value=accuracy)
		summary.value.add(tag="loss", simple_value= loss)


		# Add it to the Tensorboard summary writer
		# Make sure to specify a step parameter to get nice graphs over time
		if not train:
			self.test_writer.add_summary(summary, step)
			print('---------------------------')
			print('Test at step %s: \t accuracy:%.4f \t loss:%.4f' %(step, accuracy, loss))
		else:
			self.train_actual_writer.add_summary(summary, step)
			print('---------------------------')
			print('Training at step %s: \t actual accuracy:%.4f \t actual loss:%.4f' %(step, accuracy, loss))










