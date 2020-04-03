
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import random
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[3]:


data.test.cls = np.argmax(data.test.labels, axis=1)


# In[4]:


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


# In[5]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[6]:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[7]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[8]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[9]:


net = tf.layers.flatten(x_image)
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=256, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc2',
                      units=256, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc3',
                      units=256, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)


# In[10]:


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)


# In[11]:


loss = tf.reduce_mean(cross_entropy)


# In[12]:


opt = tf.train.AdamOptimizer(learning_rate=0.0001)
optimizer = opt.minimize(loss)


# In[13]:


y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[14]:


correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[15]:


session = tf.Session()


# In[16]:


session.run(tf.global_variables_initializer())


# In[17]:


# Counter for total number of iterations performed so far.
train_batch_size = 5500
training_loss_list=[]
testing_loss_list=[]

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(0,
                    num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        random.shuffle(y_true_batch)
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
    training_loss_list.append(los)
    
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    cls_true = data.test.cls
    
    
#    loss_testing = tf.reduce_mean(tf.abs(cls_true - y_pred_cls))
    i = 0
    while i < num_test:
        j = num_test
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
        los=session.run(loss,feed_dict=feed_dict)
    testing_loss_list.append(los)


# In[18]:


for i in range(0,2000):
    optimize(num_iterations=10)


# In[19]:


plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(training_loss_list)
plt.plot(testing_loss_list)
plt.show()

