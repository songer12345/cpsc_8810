
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


X=np.expand_dims(np.arange(0.0,1.0,0.001),1)
Y=np.sinc(5*X)
#Y=np.expand_dims([np.sin(5*np.pi*x[0])/(5*np.pi*x[0]) for x in X],1)


# In[4]:


sess=tf.Session()
x=tf.placeholder(dtype=tf.float64,shape=[1000,1])
y=tf.placeholder(dtype=tf.float64,shape=[1000,1])

input_layer=tf.layers.dense(x,1)

hidden_layer=tf.layers.dropout(input_layer,0.2)
hidden_layer2=tf.layers.dense(hidden_layer,190,activation=tf.nn.relu)


output_layer=tf.layers.dense(hidden_layer2,1)
loss=tf.losses.mean_squared_error(y,output_layer)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess.run(tf.global_variables_initializer())
loss_list=[]
for i in range(0,20000):
    _,loss_val=sess.run([train_op,loss],feed_dict={x:X,y:Y})
    loss_list.append(loss_val)
    
YP=sess.run(output_layer,feed_dict={x:X})


import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.gca().set_yscale('log')
plt.show()


# In[7]:


plt.plot(X,Y)
plt.plot(X,YP)
plt.show()


# In[18]:


sess=tf.Session()
x=tf.placeholder(dtype=tf.float64,shape=[1000,1])
y=tf.placeholder(dtype=tf.float64,shape=[1000,1])

input_layer=tf.layers.dense(x,1)
input_layer2=tf.layers.dropout(input_layer,0.2)

hidden_layer1=tf.layers.dense(input_layer2,10,activation=tf.nn.relu)
hidden_layer2=tf.layers.dense(hidden_layer1,18,activation=tf.nn.relu)
hidden_layer3=tf.layers.dense(hidden_layer2,15,activation=tf.nn.relu)
hidden_layer4=tf.layers.dense(hidden_layer3,4,activation=tf.nn.relu)

output_layer=tf.layers.dense(hidden_layer4,1)
loss=tf.losses.mean_squared_error(y,output_layer)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess.run(tf.global_variables_initializer())


loss_list2=[]
for i in range(0,20000):
    _,loss_val=sess.run([train_op,loss],feed_dict={x:X,y:Y})
    loss_list2.append(loss_val)

YP2=sess.run(output_layer,feed_dict={x:X})
import matplotlib.pyplot as plt
plt.plot(loss_list2)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.gca().set_yscale('log')
plt.show()


# In[19]:


plt.plot(X,Y)
plt.plot(X,YP2)
plt.show()


# In[27]:


sess=tf.Session()
x=tf.placeholder(dtype=tf.float64,shape=[1000,1])
y=tf.placeholder(dtype=tf.float64,shape=[1000,1])

input_layer=tf.layers.dense(x,1)
input_layer2=tf.layers.dropout(input_layer,0.3)

hidden_layer1=tf.layers.dense(input_layer,5,activation=tf.nn.relu)
hidden_layer2=tf.layers.dense(hidden_layer1,10,activation=tf.nn.relu)
hidden_layer3=tf.layers.dense(hidden_layer2,10,activation=tf.nn.relu)
hidden_layer4=tf.layers.dense(hidden_layer3,10,activation=tf.nn.relu)
hidden_layer5=tf.layers.dense(hidden_layer4,10,activation=tf.nn.relu)
hidden_layer6=tf.layers.dense(hidden_layer5,10,activation=tf.nn.relu)
hidden_layer7=tf.layers.dense(hidden_layer6,5,activation=tf.nn.relu)

output_layer=tf.layers.dense(hidden_layer7,1)
loss=tf.losses.mean_squared_error(y,output_layer)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess.run(tf.global_variables_initializer())

loss_list3=[]
for i in range(0,20000):
    _,loss_val=sess.run([train_op,loss],feed_dict={x:X,y:Y})
    loss_list3.append(loss_val)

YP3=sess.run(output_layer,feed_dict={x:X})
import matplotlib.pyplot as plt
plt.plot(loss_list3)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.gca().set_yscale('log')
plt.show()


# In[14]:


plt.plot(X,Y)
plt.plot(X,YP3)
plt.show()


# In[26]:


import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.plot(loss_list2)
plt.plot(loss_list3)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.gca().set_yscale('log')
plt.show()


# In[24]:


plt.plot(X,Y)
plt.plot(X,YP)
plt.plot(X,YP2)
plt.plot(X,YP3)
plt.show()

