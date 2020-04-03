
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


X=np.expand_dims(np.arange(0.0,1.0,0.001),1)
Y=np.sign(np.sin(5*np.pi*X))
#Y=np.expand_dims([np.sin(5*np.pi*x[0])/(5*np.pi*x[0]) for x in X],1)


# In[31]:


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


# In[32]:


plt.plot(X,Y)
plt.plot(X,YP)
plt.show()


# In[7]:


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


# In[8]:


plt.plot(X,Y)
plt.plot(X,YP2)
plt.show()


# In[26]:


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


# In[27]:


plt.plot(X,Y)
plt.plot(X,YP3)
plt.show()


# In[39]:


import csv

with open('function2_loss.csv','w',newline='') as csvfile:
    fieldnames=['times','loss1','loss2','loss3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0,20000):
        writer.writerow({'times': i, 'loss1':loss_list[i], 'loss2':loss_list2[i], 'loss3':loss_list3[i]})

with open('function2_prediction.csv','w',newline='') as csvfile:
    fieldnames=['times','GT','model1','model2','model3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0,1000):
        writer.writerow({'times': i, 'GT':Y[i], 'model1':YP[i], 'model2':YP2[i],'model3':YP3[i]})


# In[33]:


import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.plot(loss_list2)
plt.plot(loss_list3)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.gca().set_yscale('log')
plt.show()


# In[34]:


plt.plot(X,Y)
plt.plot(X,YP)
plt.plot(X,YP2)
plt.plot(X,YP3)
plt.show()

