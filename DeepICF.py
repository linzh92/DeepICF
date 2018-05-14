from __future__ import absolute_import
from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import logging

from time import time

from Dataset import Dataset
import Batch_gen as data
import Evaluate as evaluate

import argparse
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def parse_args():
    parser = argparse.ArgumentParser(description="Run FISM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--layers', nargs='?', default='[32,16]',
                        help='Size of each layer')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--reg_W', nargs='?', default='[0,0]',
                        help='L_2 regularization on each layer weights.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch norm (0 or 1)')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    return parser.parse_args()


# batch norm
def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z


class DeepICF:

    def __init__(self, num_items, args):
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.alpha = args.alpha
        self.verbose = args.verbose
        self.n_hidden = eval(args.layers)
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.reg_W = eval(args.reg_W)
        self.batch_choice = args.batch_choice
        self.train_loss = args.train_loss
        self.use_batch_norm = args.batch_norm

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # the ground truth
            self.is_train_phase = tf.placeholder(tf.bool)  # mark is training or testing

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                # why [0, 3707)?
                name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(self.num_items), name='bias')

            self.weights = {
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden[-1], 1], mean=0, stddev=np.sqrt(2.0 / (self.n_hidden[-1] + 1))),
                    name='weights_out')
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([1]), name='biases_out')
            }
            n_hidden_0 = self.embedding_size
            for i in range(len(self.n_hidden)):
                if i > 0:
                    n_hidden_0 = self.n_hidden[i - 1]
                n_hidden_1 = self.n_hidden[i]
                self.weights['h%d' % i] = tf.Variable(
                    tf.random_normal([n_hidden_0, n_hidden_1], mean=0, stddev=np.sqrt(2.0 / (n_hidden_0 + n_hidden_1))),
                    name='weights_h%d' % i)
                self.biases['b%d' % i] = tf.Variable(tf.random_normal([n_hidden_1]), name='biases_b%d' % i)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input), 1)  # (?, k)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)  # (?, k)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)  # (?, 1)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))  # (1)
            self.embedding_p = self.coeff * self.embedding_p  # (?, k)

            layer1 = tf.multiply(self.embedding_p, self.embedding_q)  # (?, k)
            for i in range(len(self.n_hidden)):
                layer1 = tf.add(tf.matmul(layer1, self.weights['h%d' % i]), self.biases['b%d' % i])
                if self.use_batch_norm:
                    layer1 = batch_norm_layer(layer1, train_phase=self.is_train_phase, scope_bn='bn_%d' % i)
                layer1 = tf.nn.relu(layer1)
            out_layer = tf.matmul(layer1, self.weights['out']) + self.biases['out']  # (?, 1)

            self.output = tf.sigmoid(tf.add_n([out_layer, self.bias_i]))  # (?, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_))

            for i in range(min(len(self.n_hidden), len(self.reg_W))):
                if self.reg_W[i] > 0:
                    self.loss = self.loss + self.reg_W[i] * tf.reduce_sum(tf.square(self.weights['h%d' % i]))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")


def training(flag, model, dataset, epochs, num_negatives):
    weight_path = 'Pretraining/%s/%s/alpha0.0.ckpt' % (model.dataset_name, model.embedding_size)
    saver = tf.train.Saver([model.c1, model.embedding_Q, model.bias])

    with tf.Session() as sess:
        # pretrain nor not
        if flag != 0:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, weight_path)
            p_c1, p_e_Q, p_b = sess.run([model.c1, model.embedding_Q, model.bias])

            model.c1 = tf.Variable(p_c1, dtype=tf.float32, trainable=True, name='c1')
            model.embedding_Q_ = tf.concat([model.c1, model.c2], 0, name='embedding_Q_')
            model.embedding_Q = tf.Variable(p_e_Q, dtype=tf.float32, trainable=True, name='embedding_Q')
            model.bias = tf.Variable(p_b, dtype=tf.float32, trainable=True, name='embedding_Q')

            logging.info("using pretrained variables")
            print("using pretrained variables")
        else:
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print("initialized")

        # initialize for training batches
        batch_begin = time()
        batches = data.shuffle(dataset, model.batch_choice, num_negatives)
        batch_time = time() - batch_begin

        num_batch = len(batches[1])
        batch_index = list(range(num_batch))

        # initialize the evaluation feed_dicts
        testDict = evaluate.init_evaluate_model(model, sess, dataset.testRatings, dataset.testNegatives,
                                                dataset.trainList)

        best_hr, best_ndcg = 0, 0
        # train by epoch
        for epoch_count in range(epochs):

            train_begin = time()
            training_batch(batch_index, model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:

                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0

                eval_begin = time()
                (hits, ndcgs, losses) = evaluate.eval(model, sess, dataset.testRatings, dataset.testNegatives, testDict)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                if hr > best_hr:
                    best_hr = hr
                    best_ndcg = ndcg

                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print(
                        "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))

            batch_begin = time()
            batches = data.shuffle(dataset, model.batch_choice, num_negatives)
            np.random.shuffle(batch_index)
            batch_time = time() - batch_begin

        return best_hr, best_ndcg


def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None], model.is_train_phase: True}
        sess.run(model.optimizer, feed_dict)


def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None], model.is_train_phase: True}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


if __name__ == '__main__':
    args = parse_args()
    regs = eval(args.regs)

    logging.info("begin training FISM model ......")
    logging.info("dataset:%s  pretrain:%d  embedding_size:%d" % (args.dataset, args.pretrain, args.embed_size))
    logging.info("regs:%.8f, %.8f  learning_rate:%.4f  train_loss:%d" % (regs[0], regs[1], args.lr, args.train_loss))

    dataset = Dataset(args.path + args.dataset)
    model = DeepICF(dataset.num_items, args)
    model.build_graph()
    best_hr, best_ndcg = training(args.pretrain, model, dataset, args.epochs, args.num_neg)
    print("End. best HR = %.4f, best NDCG = %.4f" % (best_hr, best_ndcg))

