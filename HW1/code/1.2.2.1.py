
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


# In[2]:


X=np.expand_dims(np.arange(0.0,1.0,0.001),1)
Y=np.sinc(5*X)


# In[3]:


session=tf.Session()
x=tf.placeholder(dtype=np.float32,shape=[1000,1])
y=tf.placeholder(dtype=np.float32,shape=[1000,1])


# In[4]:


net = tf.layers.dense(inputs=x, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=1, activation=None)


# In[5]:


loss=tf.losses.mean_squared_error(y,logits)


# In[6]:


opt = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = opt.minimize(loss)


# In[7]:


trainable_var_list = tf.trainable_variables()


# In[8]:


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable


# In[9]:


weights_fc1 = get_weights_variable(layer_name='layer_fc1')
weights_fc_out = get_weights_variable(layer_name='layer_fc_out')


# In[10]:


session.run(tf.global_variables_initializer())


# In[11]:


grads = tf.gradients(loss, weights_fc1)[0]
grads2 = tf.gradients(loss, weights_fc_out)[0]
grad_list=[]
grad_list2=[]
loss_list=[]


# In[15]:


# Counter for total number of iterations performed so far.


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(0,
                    num_iterations):

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
        grad_list.append(grads_vals)
        grad_list2.append(grads_vals2)


# In[16]:


optimize(num_iterations=700)


# In[17]:


from numpy import linalg as LA
grad_list_total=[]
for i in range(0,len(grad_list)):
    k=(LA.norm(grad_list[i])**2+LA.norm(grad_list2[i])**2)**0.5
    grad_list_total.append(k)


# In[18]:


plt.plot(grad_list_total)
plt.title('Model gradient')
plt.ylabel('gradient')
plt.xlabel('iteration')
plt.show()


# In[19]:


plt.plot(loss_list)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()

