import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings("ignore")
tf.set_random_seed(1)

mnist = input_data.read_data_sets('data', one_hot=True)

#hyperparas
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
	# (28, 128)
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
	# (128, 10)
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
	#(128, )
	'in': tf.Variable(tf.constant(0.1,shape=[n_hidden_units, ])),
	#(10, )
	'out': tf.Variable(tf.constant(0.1,shape=[n_classes, ]))
}

def RNN(X, weights, biases):
	# hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
	# X ==> (128 batch * 28 steps, 28 inputs)
	X = tf.reshape(X, [-1, n_inputs])
	
	# into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
	X_in = tf.matmul(X, weights['in']) + biases['in']
	
	# X_in ==> (128 batch, 28 steps, 128 hidden)
	X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
	
	# cell
    ##########################################
	cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
	# lstm cell is divided into two parts (c_state, h_state) #batch_size =128
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	# dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
	outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
	# outputs = (128 batch, 28 steps, 128 outputs)
	# unpack to list [(batch, outputs)..] * steps
	outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
	# results (128 batch, 10 class)
	results = tf.matmul(outputs[-1], weights['out']) + biases['out']
	return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	step = 0
	while step * batch_size < training_iters:
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
		sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys, })
		if step % 20 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, })
			print ("step %d, training accuracy %g"%(step, train_accuracy))
		step += 1
	