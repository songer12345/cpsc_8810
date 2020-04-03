
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
from sklearn.utils import shuffle
import pickle
import json
import re
import subprocess


# In[2]:


source_path='dataset/training_data/feat/'
target_path='dataset/training_label.json'


# In[2]:


def load_data(path):
    filenames=os.listdir(path)
    filenames.sort()
    data=[]
    for i in filenames:
        data.append(np.load(path+i))
    return data


# In[4]:


source=load_data(source_path)


# In[5]:


target=json.load(open(target_path))

target={i['id']:i['caption'] for i in target}

target=[target[i] for i in sorted(target.keys())]


# In[6]:


'''
target2=[]
for i in target:
    test=[]
    for j in i:
        test.append(j.lower())
    test.sort()
    target2.append(test[0:10])
'''


# In[7]:


'''target2=[]
for j in target:
    for k in j:
        if len(k.lower().split())<3:
            j.remove(k)

    a=[i.lower().split()[0] for i in j]
    b=[i.lower().split()[1] for i in j]
    c=[i.lower().split()[2] for i in j]

    a_most=max(set(a),key=a.count)
    b_most=max(set(b),key=b.count)
    c_most=max(set(c),key=c.count)
    
    test=[]
    
    for i in range(0,len(j)):
        if a[i]==a_most and b[i]==b_most and c[i]==c_most:
            test.append(j[i])
    
    if test==[]:
        test=[j[0]]
    target2.append(test)
'''


# In[8]:


sentences=[]
for k in target:
    words=[]
    sentence=[]
    for j in k:
        if len(j.lower().split())>3:
            words.append(j.lower().split())
        
    a=[i[0] for i in words]
    b=[i[1] for i in words]
    c=[i[2] for i in words]

    for i in range(0,len(words)):
        for j in range(i,len(words)):
            if a.count(words[i][0])<a.count(words[j][0]):
                l=words[i]
                words[i]=words[j]
                words[j]=l

    for i in range(0,len(words)):
        for j in range(i,len(words)):
            if b.count(words[i][1])<b.count(words[j][1]) and a.count(words[i][0])<=a.count(words[j][0]):
                l=words[i]
                words[i]=words[j]
                words[j]=l

    for i in range(0,len(words)):
        for j in range(i,len(words)):
            if c.count(words[i][2])<c.count(words[j][2]) and a.count(words[i][0])<=a.count(words[j][0]) and b.count(words[i][1])<=b.count(words[j][1]):
                l=words[i]
                words[i]=words[j]
                words[j]=l

    for i in words[0:10]:
        sentence.append(' '.join(i))
    sentences.append(sentence)


# In[9]:


target=sentences


# In[10]:


vocab_to_int={'<PAD>':0,'<EOS>':1,'<BOS>':2,'<UNK>':3}
num=4


# In[11]:


target_int=[]

for i in target:
    sentences_to_int=[]
    for sentence in i:
        words=res = re.findall(r'\w+', sentence.lower())
        sentence_to_int=[]
        for word in words:
            if (word in vocab_to_int)==False:
                vocab_to_int[word]=num
                num+=1
            sentence_to_int.append(vocab_to_int[word])
        sentences_to_int.append(sentence_to_int)
    target_int.append(sentences_to_int)


# In[12]:


int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}


# In[13]:


source_text=[]
target_text=[]

for i in range(0,len(source)):
    for j in range(0,len(target_int[i])):
        source_text.append(source[i])
        target_text.append(target_int[i][j])


# In[14]:


pickle.dump((vocab_to_int,int_to_vocab,source_text,target_text),open('preprocess.p','wb'))


# In[3]:


def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


# In[4]:


vocab_to_int,int_to_vocab,source_text,target_text=load_preprocess()


# In[5]:


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.float32, [None,80,4096], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len


# In[6]:


def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob


# In[7]:


def process_decoder_input(target_data, target_vocab_to_int, batch_size):

    bos_id = target_vocab_to_int['<BOS>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], bos_id), after_slice], 1)
    
    return after_concat


# In[8]:


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size):
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                       rnn_inputs,
                                       dtype=tf.float32)
    return outputs, state


# In[24]:


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, dec_embeddings,
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
#    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
#                                               target_sequence_length)
#    
#    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
#                                                      tf.fill([batch_size], start_of_sequence_id), 
#                                                      end_of_sequence_id)
    
    helper=tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(dec_embed_input,
                                                              target_sequence_length,
                                                             dec_embeddings,
                                                             0.5)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs


# In[10]:


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):

    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


# In[11]:


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):

    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            dec_embeddings,
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<BOS>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)


# In[12]:


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):

    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = decoding_layer(dec_input,
                                               enc_states, 
                                               target_sequence_length, 
                                               max_target_sentence_length,
                                               rnn_size,
                                              num_layers,
                                              target_vocab_to_int,
                                              target_vocab_size,
                                              batch_size,
                                              keep_prob,
                                              dec_embedding_size)
    
    return train_output, infer_output


# In[13]:


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size,target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""

    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch=np.array(sources_batch)
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        length=len(pad_targets_batch)
        target_length=len(pad_targets_batch[0])
        
        for i in range(0,length):
            pad_targets_lengths.append(target_length)
            
#        for target in pad_targets_batch:
#            pad_targets_lengths.append(len(target))
        
#        print(pad_targets_lengths)
        yield pad_sources_batch, pad_targets_batch, pad_targets_lengths


# In[14]:


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


# In[15]:


epochs = 20
batch_size = 64

rnn_size = 128
num_layers = 2

decoding_embedding_size = 500

learning_rate = 0.001
keep_probability = 0.7


# In[25]:


save_path = 'checkpoints/dev'

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()
    
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_text),
                                                   len(vocab_to_int),
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   vocab_to_int)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        optimizer = tf.train.AdamOptimizer(lr)
        
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# In[17]:


train_source = source_text[batch_size:]
train_target = target_text[batch_size:]
valid_source = source_text[:batch_size]
valid_target = target_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_targets_lengths ) = next(get_batches(valid_source,valid_target,batch_size,vocab_to_int['<PAD>']))                                                                                                  


# In[18]:


filenames=os.listdir('dataset/testing_data/feat/')

filenames.sort()

target_path='dataset/testing_label.json'

target_test=json.load(open(target_path))

target_test={i['id']:i['caption'] for i in target_test}

target_test=[target_test[i] for i in sorted(target_test.keys())]

source_test_path='dataset/testing_data/feat/'
source_test=load_data(source_test_path)


# In[19]:


cd /home/liao5/dataset


# In[20]:


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


# In[21]:


maxpoint=0

def model_testing():
    result=[]
    k=0
    logits=inference_logits
    for i in range(0,len(source_test)):
        translate_logits = sess.run(logits, {input_data: [source_test[i]]*batch_size,
                                             target_sequence_length: [20]*batch_size,
                                             keep_prob: 1.0})[0]
        result.append(translate_logits)
        k=k+1

    filenames=os.listdir('testing_data/feat/')
    filenames.sort()
    f=open('../result.txt','w')
    for i in range(0,len(result)):
        f.write(filenames[i][:-4])
        f.write(',')
        f.write(" ".join([int_to_vocab[j] for j in result[i]]).replace(' <PAD>',''))
        f.write('\n')
    f.close()

    proc = subprocess.Popen(['python', 'bleu_eval.py',  '../result.txt'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    point= float(proc.communicate()[0].split()[-1])
    print(point)
    
    global maxpoint
    if point > maxpoint:
        maxpoint = point
        f=open('../result_max_'+str(maxpoint)+'.txt','w')
        for i in range(0,len(result)):
            f.write(filenames[i][:-4])
            f.write(',')
            f.write(" ".join([int_to_vocab[j] for j in result[i]]).replace(' <PAD>',''))
            f.write('\n')
        f.close()
        print('max point is:'+  str(maxpoint))


# In[ ]:


display_step=int(len(source_text) // batch_size/3-1)

epochs=int(1450*200/len(source_text))
#epochs=200

print("epochs: "+ str(epochs)+ " times")

with tf.Session(graph=train_graph) as sess:

    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        train_source,train_target=shuffle(train_source,train_target)
        for batch_i, (source_batch, target_batch, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)
                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_text) // batch_size, train_acc, valid_acc, loss))

        if epoch_i%1==0:
            model_testing()

    saver = tf.train.Saver()
    saver.save(sess, save_path)
    save_params(save_path)
    print('Model Trained and Saved')

