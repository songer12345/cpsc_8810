import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    

def read_image():
    f=open('dataset/data_batch_'+str(1),'rb')
    data=pickle.load(f,encoding='bytes')
    image_input=data[b'data']
    for i in range(2,6):
        f=open('dataset/data_batch_'+str(i),'rb')
        data=pickle.load(f,encoding='bytes')
        image_input=np.concatenate((image_input,data[b'data']),axis=0)
        f.close()
    return image_input


image_input=read_image()


image_input2=[]
for i in image_input:
    image_input2.append(np.reshape(i.reshape(3,32,32),3072,order='F'))

image_input3=np.array(image_input2).reshape(50000,32,32,3)

session = tf.Session()
image_input4=session.run(tf.image.resize_images(image_input3,(64,64)))

image_input=image_input4/255



noise = tf.placeholder(tf.float32, shape=[None, 100])

image = tf.placeholder(tf.float32, shape=[None,64,64,3])

def generator(x):
    activation = tf.nn.relu
    with tf.variable_scope("generator",reuse=None):
        net = tf.layers.dense(inputs=x,units=4*4*512, activation=activation)
        net = tf.layers.dropout(net, 0.8)
        net = tf.contrib.layers.batch_norm(net, decay=0.9)

        net = tf.reshape(net, shape=[-1,4,4,512])

        net = tf.layers.conv2d_transpose(inputs=net,strides=2,padding='same',
                               filters=256, kernel_size=5, activation=activation)
        net = tf.layers.dropout(net, 0.8)
        net = tf.contrib.layers.batch_norm(net, decay=0.9)

        net = tf.layers.conv2d_transpose(inputs=net,strides=2,padding='same',
                               filters=128, kernel_size=5, activation=activation)
        net = tf.layers.dropout(net, 0.8)
        net = tf.contrib.layers.batch_norm(net, decay=0.9)

        net = tf.layers.conv2d_transpose(inputs=net,strides=2,padding='same',
                               filters=64, kernel_size=5, activation=activation)
        net = tf.layers.dropout(net, 0.8)
        net = tf.contrib.layers.batch_norm(net, decay=0.9)

        net = tf.layers.conv2d_transpose(inputs=net,strides=2,padding='same',
                               filters=3, kernel_size=5, activation=tf.nn.sigmoid)
        
        x = tf.nn.tanh(x)

        return net



def discriminator(image_input,reuse=None):

    with tf.variable_scope("discriminator", reuse=reuse):

        image_input=tf.reshape(image_input,shape=[-1,64,64,3])

        net = tf.layers.conv2d(inputs=image_input, strides=2,padding='same',
                               filters=64, kernel_size=5, activation=tf.nn.leaky_relu)
        
        net = tf.layers.conv2d(inputs=net, strides=2,padding='same',
                               filters=128, kernel_size=5, activation=tf.nn.relu)

        net=tf.contrib.layers.batch_norm(net, decay=0.9,activation_fn=tf.nn.leaky_relu)

        net = tf.layers.conv2d(inputs=net, strides=2,padding='same',
                               filters=256, kernel_size=5, activation=tf.nn.relu)

        net=tf.contrib.layers.batch_norm(net, decay=0.9,activation_fn=tf.nn.leaky_relu)

        net = tf.layers.conv2d(inputs=net, strides=2,padding='same',
                               filters=512, kernel_size=5, activation=tf.nn.relu)

        net=tf.contrib.layers.batch_norm(net, decay=0.9)

        net = tf.layers.conv2d_transpose(inputs=net,strides=(1,1),
                               filters=512, kernel_size=1, activation=tf.nn.relu)

        net=tf.layers.flatten(net)

        net = tf.layers.dense(inputs=net,units=1, activation='sigmoid')

        return net



def gan_loss_function(x,y):
    eps=1e-12
    return (-(x*tf.log(y+eps)+(1-x)*tf.log(1-y+eps)))



g = generator(noise)

d_real=discriminator(image)
d_fake=discriminator(g,reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)


loss_d_real=gan_loss_function(tf.ones_like(d_real),d_real)
loss_d_fake=gan_loss_function(tf.zeros_like(d_fake),d_fake)
loss_g=tf.reduce_mean(gan_loss_function(tf.ones_like(d_fake),d_fake))
loss_d=tf.reduce_mean(0.5*(loss_d_real+loss_d_fake))


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.00015, beta1=0.5).minimize(loss_d , var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.00015, beta1=0.5).minimize(loss_g , var_list=vars_g)


session = tf.Session()
session.run(tf.global_variables_initializer())



batch_size=64
dataset_size=len(image_input)


j=0
epoch=100
for i in range(0,epoch * 781):
    train_g = True
    train_d = True

    if j+batch_size > dataset_size:
        j=0
        print("epoch:"+str(int(i/781))+", d_loss:"+str(d_loss)+", g_loss:"+str(g_loss))
        continue
    else:
        batch=image_input[j:j+batch_size]
        j=j+batch_size

    noise_input=np.random.random_sample((len(batch),100))
    
    d_real_loss, d_fake_loss, g_loss, d_loss=session.run([loss_d_real, loss_d_fake, loss_g, loss_d],
                                                        feed_dict={noise: noise_input, image:batch})
    fake_image=session.run([g],feed_dict={noise: noise_input, image:batch})
    
    d_real_loss=np.mean(d_real_loss)
    d_fake_loss=np.mean(d_fake_loss)

    if g_loss * 1.5< d_loss:
        train_g = False

    if d_loss * 2< g_loss:
        train_d = False

    if train_d:
        session.run(optimizer_d, feed_dict={noise: noise_input, image:batch})

    if train_g:       
        session.run(optimizer_g, feed_dict={noise: noise_input, image:batch})

    if i%100==0:
        print("iteration:"+str(i)+", d_loss:"+str(d_loss)+", g_loss:"+str(g_loss))


fake_image=session.run([g],feed_dict={noise: noise_input, image:batch})



import matplotlib.pyplot as plt
plt.imshow(fake_image[0][0])
plt.show()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))



print(device_lib.list_local_devices())




tf.test.is_gpu_available()



with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

