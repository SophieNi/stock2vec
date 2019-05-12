import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
import numpy as np
import multiprocessing
from gensim.models import Word2Vec

def skip2gram(filepath):
	df = pd.read_csv(filepath)
	# get top 5 correlated tickers
	df['Top 20 tickers'] = df['Top 20 tickers'].map(lambda x: str(eval(x)[:5]))
	df['Top 20 cor vals'] = df['Top 20 cor vals'].map(lambda x: str(eval(x)[:5]))
	df = df[df['Top 20 tickers'] != "[]"]
	corpus = df.apply(lambda x : [x['Ticker']] + eval(x['Top 20 tickers']), axis=1).to_numpy()

	# count the number of cores in a computer
	cores = multiprocessing.cpu_count()
	# run word2vex model
	w2v_model = Word2Vec(min_count=20, window=6, size=5, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores-1)
	w2v_model.build_vocab(corpus)
	w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=30)
	w2v_model.init_sims(replace=True)

	# test the top 20 most correlated tickers for 'BMO' ticker
	2v_model.wv.most_similar(positive=['BMO'], topn=20)
	
	# save all the unique ticker names in word_index.txt, this is metadata
	outf = open('word_index.txt', 'w')
    outf.write("\n".join(w2v_model.wv.index2word))
    # save the vector for each ticker corresponding in word_index.txt in word_vectors.txt
	np.savetxt('word_vectors.txt', w2v_model.wv.vectors)

skip2gram(filepath)

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np

# procedures to start tensorboard session
def runTensorBoard():
    embedding = np.empty((len(w2v_model.wv.index2word), 5), dtype=np.float32)
    for i, vector in enumerate(w2v_model.wv.vectors):
        embedding[i] = vector
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})
    
    summary_writer = tf.summary.FileWriter('log', sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)
    
    # save the model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('log', "model.ckpt"))

# to open tensorboard visualize data, on terminal run
# tensorboard --logdir=log
