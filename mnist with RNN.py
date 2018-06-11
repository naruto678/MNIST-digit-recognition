import tensorflow as tf 	
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#params
time_steps=28
batch_size=128
element_size=28
hidden_layer_size=128
epochs=2
num_classes=10
batch_iter=1000
 #end of params
 #utils
import numpy as np  
def vectorize(sequences,dims=10):
	res=np.zeros((len(sequences),dims))
	for i,sequence in enumerate(sequences):
		res[i,sequence]=1
	return res.astype('float32')
		
x_train=x_train.astype('float32')
y_train=vectorize(y_train)
x_test=x_test.astype('float32')
y_test=vectorize(y_test)

x_place=tf.placeholder(tf.float32,[None,time_steps,element_size])
y_place=tf.placeholder(tf.float32,[None,num_classes])

with tf.name_scope('rnn_weights') as scope:
	wl=tf.Variable(tf.zeros((element_size,hidden_layer_size)))
	wh=tf.Variable(tf.zeros((hidden_layer_size,hidden_layer_size)))
	bl=tf.Variable(tf.zeros((hidden_layer_size)))


def rnn_step(previous_hidden_layer,x):
	initial=tf.tanh(tf.matmul(x,wl)+tf.matmul(previous_hidden_layer,wh)+bl)
	return initial
processed_inputs=tf.transpose(x_place,perm=[1,0,2])
print('processed_inputs ',processed_inputs.get_shape())
initial_hidden=tf.zeros((batch_size,hidden_layer_size))
all_hidden_states=tf.scan(rnn_step,processed_inputs,initializer=initial_hidden)
print('all_hidden_states',all_hidden_states.get_shape())
with tf.name_scope('linear_weights') as scope:
	WL=tf.Variable(tf.truncated_normal((hidden_layer_size,num_classes)))
	BL=tf.Variable(tf.truncated_normal([num_classes]))
def linear_layer(hidden_state):
	return tf.matmul(hidden_state,WL)+BL
all_outputs=tf.map_fn(linear_layer,all_hidden_states)
print('all_outputs',all_outputs.get_shape())
output=all_outputs[-1]
print(output.get_shape())
with tf.name_scope('accuracy'):
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y_place))
	train_step=tf.train.RMSPropOptimizer(0.001,0.9).minimize(loss)
	accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(y_place,1)),tf.float32))
# finally the training part
import random
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epochs):
		for j in range(batch_iter):
			ids=random.sample(range(len(x_train)),batch_size)
			partial_x_train=x_train[ids]
			partial_y_train=y_train[ids]

			if j>0 and j%100==0:
				training_loss=sess.run(loss,{x_place:partial_x_train,y_place:partial_y_train})
				print('epoch-'+str(i)+' iter- '+str(j)+' training_loss- '+str(training_loss))
			sess.run(train_step,{x_place:partial_x_train,y_place:partial_y_train})

	# at the end of each epoch print the testing accuracy
		testing_accuracy=sess.run(accuracy,{x_place:x_test[:128],y_place:y_test[:128]})
		print('testing accuracy ',testing_accuracy)



	





