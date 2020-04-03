
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[3]:


data.test.cls = np.argmax(data.test.labels, axis=1)


# In[4]:


img_size = 28

img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_channels = 1

num_classes = 10


# In[5]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[6]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[7]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[8]:


def setgraph(logits):
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    global loss 
    loss= tf.reduce_mean(cross_entropy)
    
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    global accuracy 
    accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    global optimizer 
    optimizer= opt.minimize(loss)
    
    global gradient
    gradient = tf.gradients(loss, x)[0]


# In[9]:


def optimize(num_iterations,train_batch_size):

    global total_iterations

    for i in range(0,
                    num_iterations):
        
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        session.run(optimizer, feed_dict=feed_dict_train)
        
        los=session.run(loss,feed_dict=feed_dict_train)
        acc=session.run(accuracy,feed_dict=feed_dict_train)
        grad=session.run(gradient,feed_dict=feed_dict_train)
        
    loss_training_list.append(los)
    accu_training_list.append(acc)
    grad_list.append(grad)


# In[10]:


test_batch_size = 256
def print_test_accuracy():

    num_test = len(data.test.images)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    i = 0

    while i < num_test:
        
        j = num_test

        images = data.test.images[i:j, :]

        labels = data.test.labels[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls


    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test
    los=session.run(loss, feed_dict=feed_dict)
    loss_testing_list.append(los)
    accu_testing_list.append(acc)


# In[11]:


net = tf.layers.dense(inputs=x,
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net,
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net,
                      units=num_classes, activation=None)

setgraph(logits)


# In[12]:


loss_training_list=[]
accu_training_list=[]
loss_testing_list=[]
accu_testing_list=[]
grad_list=[]


# In[13]:


## model 0
batch=[10,33,100,333,1000]

for j in range(0,5):
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    optimize(10000,batch[j])
    print_test_accuracy()


# In[14]:


sensitive=[]
from numpy import linalg as LA
for i in grad_list:
    sensitive.append(LA.norm(i))


# In[15]:


fig, ax1 = plt.subplots()

ax1.plot(batch,accu_training_list , marker='o', linestyle='-', color='b', label='train')
ax1.plot(batch,accu_testing_list , marker='o', linestyle='--', color='b', label='test')
ax1.set_xlabel('batch size')
ax1.set_ylabel('accu') 
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(batch,sensitive , marker='o', linestyle='-', color='r', label='sensitivity')
ax2.set_ylabel('sensitive') 

plt.title('accuracy with sensitive')
plt.gca().set_xscale('log')
plt.legend()
plt.show()


# In[16]:


fig, ax1 = plt.subplots()


ax1.plot(batch,loss_training_list , marker='o', linestyle='-', color='b', label='train')
ax1.plot(batch,loss_testing_list , marker='o', linestyle='--', color='b', label='test')
ax1.set_xlabel('batch size')
ax1.set_ylabel('loss') 
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(batch,sensitive , marker='o', linestyle='-', color='r', label='sensitivity')
ax2.set_ylabel('sensitive') 
ax2.legend(loc='upper right')

plt.title('loss with sensitive')
plt.gca().set_xscale('log')
plt.show()

