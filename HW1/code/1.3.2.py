
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


def setlayer(model):
    net = tf.layers.flatten(x_image)
    net = tf.layers.dense(inputs=net,
                          units=model[0], activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net,
                          units=model[1], activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, 
                          units=model[2], activation=tf.nn.relu)    
    logits = tf.layers.dense(inputs=net,
                          units=num_classes, activation=None)
    return logits


# In[10]:


# Counter for total number of iterations performed so far.
train_batch_size = 55

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
    loss_training_list.append(los)
    accu_training_list.append(acc)


# In[11]:


# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        los=session.run(loss,feed_dict=feed_dict)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    loss_testing_list.append(los)
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.ß
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test
    accu_testing_list.append(acc)
    # Print the accuracy.
#    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#    print(msg.format(acc, correct_sum, num_test))


# In[12]:


models=[0]*11
models[0]=[128,128,128]
models[1]=[128,128,256]
models[2]=[128,256,256]
models[3]=[256,256,256]
models[4]=[256,256,512]
models[5]=[256,512,512]
models[6]=[512,512,512]
models[7]=[512,512,1024]
models[8]=[512,1024,1024]
models[9]=[1024,1024,1024]
models[10]=[1024,1024,2048]


# In[13]:


loss_training_list=[]
accu_training_list=[]
loss_testing_list=[]
accu_testing_list=[]


# In[14]:


for model in models:
    print(model)
    logits=setlayer(model)
#    print(logits)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = opt.minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    optimize(num_iterations=10000)
    print_test_accuracy()


# In[15]:


parameters=[]
for i in range(0,11):
    array=[784,models[i][0],models[i][1],models[i][2],10]
    k=0
    for j in range(0,len(array)-1):
        k=k+(array[j]+1)*array[j+1]
    parameters.append(k)


# In[16]:


import matplotlib.pyplot as plt

plt.scatter(parameters, loss_training_list)
plt.scatter(parameters, loss_testing_list)

plt.title("", fontsize=19)
plt.xlabel("number of parameters", fontsize=10)
plt.ylabel("loss", fontsize=10)
plt.xticks(np.arange(0, 4000001, 1000000)) 
plt.show()


# In[17]:


import matplotlib.pyplot as plt

plt.scatter(parameters, accu_training_list)
plt.scatter(parameters, accu_testing_list)

plt.title("", fontsize=19)
plt.xlabel("minimum_ratio", fontsize=10)
plt.ylabel("loss", fontsize=10)
plt.xticks(np.arange(0, 4000001, 1000000)) 
plt.show()

