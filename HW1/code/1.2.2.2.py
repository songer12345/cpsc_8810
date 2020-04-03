
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


# In[5]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[6]:


data.test.cls = np.argmax(data.test.labels, axis=1)


# In[7]:


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[8]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[9]:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[10]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[11]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[12]:


net = tf.layers.flatten(x_image)
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc2',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)


# In[13]:


y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[14]:


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)


# In[15]:


loss = tf.reduce_mean(cross_entropy)


# In[16]:


opt = tf.train.AdamOptimizer(learning_rate=1e-4)
optimizer = opt.minimize(loss)


# In[17]:


correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[18]:


trainable_var_list = tf.trainable_variables()


# In[19]:


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable


# In[20]:


weights_fc1 = get_weights_variable(layer_name='layer_fc1')
weights_fc2 = get_weights_variable(layer_name='layer_fc2')
weights_fc_out = get_weights_variable(layer_name='layer_fc_out')


# In[38]:


session = tf.Session()


# In[39]:


session.run(tf.global_variables_initializer())


# In[40]:


grads = tf.gradients(loss, weights_fc1)[0]
grads2 = tf.gradients(loss, weights_fc2)[0]
grads3 = tf.gradients(loss, weights_fc_out)[0]
grad_list=[]
grad_list2=[]
grad_list3=[]


# In[41]:


hessian = tf.reduce_sum(tf.hessians(loss, weights_fc_out)[0], axis = 2)


# In[42]:


train_batch_size = 55
loss_list=[]


# In[43]:


# Counter for total number of iterations performed so far.


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(0,
                    num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
#        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
        los, acc = session.run([loss, accuracy], feed_dict=feed_dict_train)
        loss_list.append(los)
        grads_vals, grads_vals2,grads_vals3 = session.run([grads, grads2, grads3], feed_dict=feed_dict_train)
        grad_list.append(grads_vals)
        grad_list2.append(grads_vals2)
        grad_list3.append(grads_vals3)


# In[44]:


optimize(num_iterations=17500)


# In[45]:


from numpy import linalg as LA
grad_list_total=[]
for i in range(0,len(grad_list)):
    k=(LA.norm(grad_list[i])**2+LA.norm(grad_list2[i])**2+LA.norm(grad_list3[i])**2)**0.5
    grad_list_total.append(k)


# In[46]:


plt.plot(grad_list_total)
plt.title('Model gradient')
plt.ylabel('gradient')
plt.xlabel('iteration')
plt.show()


# In[47]:


plt.plot(loss_list)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()

