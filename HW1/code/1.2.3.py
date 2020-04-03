
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from numpy import linalg as LA
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


# In[2]:


X=np.expand_dims(np.arange(0.0,1.0,0.001),1)
Y=np.sinc(5*X)


# In[3]:


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable


# In[4]:


x=tf.placeholder(dtype=np.float32,shape=[1000,1])
y=tf.placeholder(dtype=np.float32,shape=[1000,1])

net = tf.layers.dense(inputs=x, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc2',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=1, activation=None)

loss=tf.losses.mean_squared_error(y,logits)

opt = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = opt.minimize(loss)

trainable_var_list = tf.trainable_variables()

weights_fc1 = get_weights_variable(layer_name='layer_fc1')
weights_fc_out = get_weights_variable(layer_name='layer_fc_out')


# In[22]:


# Counter for total number of iterations performed so far.
from numpy import linalg as LA
def optimize():
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(0,1000):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
#         _,loss_val=sess.run([train_op,loss],feed_dict={x:X,y:Y})

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        _,los=session.run([optimizer,loss], feed_dict={x:X,y:Y})
#       _,loss_val=sess.run([train_op,loss],feed_dict={x:X,y:Y})
        # Print status every 100 iterations.
#        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
#        los, acc = session.run([loss, accuracy], feed_dict=feed_dict_train)
        loss_list.append(los)
        grads_vals, grads_vals2 = session.run([grads, grads2], feed_dict={x:X,y:Y})
        grad_list.append(LA.norm([LA.norm(grads_vals),LA.norm(grads_vals2)]))
        fc1_list.append(session.run(weights_fc1))
        fc_out_list.append(session.run(weights_fc_out))


# In[33]:


result=[]
result_loss=[]
for i in range(0,100):
    print(i)
    loss_list=[]
    grad_list=[]
    fc1_list=[]
    fc_out_list=[]
    session=tf.Session()
    session.run(tf.global_variables_initializer())
    optimizer = opt.minimize(loss)
    grads = tf.gradients(loss, weights_fc1)[0]
    grads2 = tf.gradients(loss, weights_fc_out)[0]
    optimize()
    grad=grads**2+grads2**2
    optimizer = opt.minimize(grad)
    optimize()
    domain=50
    norm=LA.norm([LA.norm(fc1_list[2000-domain]),LA.norm(fc_out_list[2000-domain])])
    k=0
    for j in range(2000-2*domain,2000):
        if LA.norm([LA.norm(fc1_list[j]),LA.norm(fc_out_list[j])])>norm:
            k=k+1
    result.append(k/(2*domain))
    result_loss.append(loss_list[2000-domain])


# In[34]:


import matplotlib.pyplot as plt


plt.scatter(result, result_loss)

plt.title("", fontsize=19)
plt.xlabel("minimum_ratio", fontsize=10)
plt.ylabel("loss", fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.show()

