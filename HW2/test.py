import tensorflow as tf
import os
import pickle
import numpy as np
import sys


test_data_path,output_path=sys.argv[1:3]

def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

load_path = load_params()

def load_data(path):
    filenames=os.listdir(path)
    filenames.sort()
    data=[]
    for i in filenames:
        data.append(np.load(path+'/'+i))
    return data

source_test_path=test_data_path
source_test=load_data(source_test_path)

result=[]
loaded_graph = tf.Graph()
k=0
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    for i in range(0,len(source_test)):
        translate_logits = sess.run(logits, {input_data: [source_test[i]]*64,
                                             target_sequence_length: [20]*64,
                                             keep_prob: 1.0})[0]
        result.append(translate_logits)
        k=k+1

filenames=os.listdir(test_data_path)

filenames.sort()

def load_preprocess():
    with open('preprocess_int_to_vocab.p', mode='rb') as in_file:
        return pickle.load(in_file)
int_to_vocab=load_preprocess()

f=open(output_path,'w')
for i in range(0,len(result)):
    f.write(filenames[i][:-4])
    f.write(',')
    f.write(" ".join([int_to_vocab[j] for j in result[i]]).replace(' <PAD>',''))
    f.write('\n')
f.close()
print('finished')