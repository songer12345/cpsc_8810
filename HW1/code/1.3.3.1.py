
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


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[7]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[8]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[9]:


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
        bias=tf.get_variable('bias')
    return variable, bias


# In[10]:


kernel=0
bias=0
layer=[kernel,bias]
layers=[layer]*3
model=[layers]*3


# In[11]:


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


# In[12]:


loss_list=[]
accu_list=[]


# In[13]:


def optimize(num_iterations, modelnumber, train_batch_size):

    global total_iterations

    for i in range(0,
                    num_iterations):
        
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        session.run(optimizer, feed_dict=feed_dict_train)
        
        los=session.run(loss,feed_dict=feed_dict_train)
        acc=session.run(accuracy,feed_dict=feed_dict_train)
        
        loss_list.append(los)
        accu_list.append(acc)
    weights=[]
    weights.append(session.run(model[modelnumber][0]))
    weights.append(session.run(model[modelnumber][1]))
    weights.append(session.run(model[modelnumber][2]))
    
    return weights


# In[14]:


## model 0
net = tf.layers.flatten(x_image)
net = tf.layers.dense(inputs=net, name='layer_fc01',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc02',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out0',
                      units=num_classes, activation=None)
setgraph(logits)


# In[15]:


model[0][0] = get_weights_variable(layer_name='layer_fc01')
model[0][1] = get_weights_variable(layer_name='layer_fc02')
model[0][2] = get_weights_variable(layer_name='layer_fc_out0')


# In[16]:


session = tf.Session()
session.run(tf.global_variables_initializer())

weights0=optimize(50000,0,64)


# In[17]:


## model 1
net = tf.layers.flatten(x_image)
net = tf.layers.dense(inputs=net, name='layer_fc11',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc12',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out1',
                      units=num_classes, activation=None)

setgraph(logits)


# In[18]:


model[1][0] = get_weights_variable(layer_name='layer_fc11')
model[1][1] = get_weights_variable(layer_name='layer_fc12')
model[1][2] = get_weights_variable(layer_name='layer_fc_out1')


# In[19]:


session = tf.Session()
session.run(tf.global_variables_initializer())

weights1=optimize(10000,1,1024)


# In[20]:


test_batch_size=256

def print_train_accuracy():

    num_train = len(data.train.images)

    cls_pred = np.zeros(shape=num_train, dtype=np.int)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    i = 0

    while i < num_train:
        
        j = min(i + test_batch_size, num_train)

        images = data.train.images[i:j, :]

        labels = data.train.labels[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = session.run(tf.argmax(data.train.labels, dimension=1))

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_train
    accu_training_list.append(acc)
    
    images = data.train.images[0:10000, :]
    labels = data.train.labels[0:10000, :]
    feed_dict = {x: images,
                 y_true: labels}
    
    los=session.run(loss, feed_dict=feed_dict)
    loss_training_list.append(los)




def print_test_accuracy():

    num_test = len(data.test.images)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    i = 0

    while i < num_test:
        
        j = min(i + test_batch_size, num_test)

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
    accu_testing_list.append(acc)
    
    images = data.test.images[0:10000, :]
    labels = data.test.labels[0:10000, :]
    feed_dict = {x: images,
                 y_true: labels}
    los=session.run(loss, feed_dict=feed_dict)
    loss_testing_list.append(los)


# In[21]:


# model 2
net = tf.layers.flatten(x_image)
net = tf.layers.dense(inputs=net, name='layer_fc21',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc22',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out2',
                      units=num_classes, activation=None)

setgraph(logits)


# In[22]:


# calculate
alphas=[]

loss_training_list=[]
accu_training_list=[]

loss_testing_list=[]
accu_testing_list=[]


kernel=[0]*3
bias=[0]*3
for a in range(0,7):
    alpha=a*0.5-1.0
    alphas.append(alpha)
    for i in range(0,3):
        kernel[i]=tf.Variable(weights0[i][0]*(1-alpha)+weights1[i][0]*alpha)
        bias[i]=tf.Variable(weights0[i][1]*(1-alpha)+weights1[i][1]*alpha)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    model[2][0]= get_weights_variable(layer_name='layer_fc21')
    model[2][1]= get_weights_variable(layer_name='layer_fc22')
    model[2][2]= get_weights_variable(layer_name='layer_fc_out2')

    session.run(model[2][0][0].assign(kernel[0]))
    session.run(model[2][1][0].assign(kernel[1]))
    session.run(model[2][2][0].assign(kernel[2]))

    session.run(model[2][0][1].assign(bias[0]))
    session.run(model[2][1][1].assign(bias[1]))
    session.run(model[2][2][1].assign(bias[2]))

    print_train_accuracy()
    print_test_accuracy()
#session.close()


# In[33]:


fig, ax1 = plt.subplots()

ax1.plot(alphas,loss_training_list , marker='o', linestyle='-', color='b', label='train')
ax1.plot(alphas,loss_testing_list , marker='o', linestyle='--', color='b', label='test')
ax1.set_xlabel('alpha')
ax1.set_ylabel('loss',color='b') 
ax1.legend(loc='upper left',prop={'size': 8})
ax1.set_ylim([0,2])

ax2 = ax1.twinx()
ax2.plot(alphas,accu_training_list , marker='o', linestyle='-', color='r', label='train')
ax2.plot(alphas,accu_testing_list , marker='o', linestyle='--', color='r', label='test')
ax2.set_ylabel('accuracy',color='r') 
ax2.legend(loc='upper right',prop={'size':8})

plt.title('loss and accuracy')
plt.show()


# In[24]:


loss_training_list


# In[25]:


loss_testing_list

