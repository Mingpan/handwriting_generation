
import os
import time
import math
import random

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# the nest package has changed its location
if tf.__version__.startswith("1.2"):
  from tensorflow.python.util import nest
elif tf.__version__.startswith("1.4"):
  from tensorflow.contrib.framework import nest
else:
  raise ValueError("Cannot locate tensorflow 'nest' package!")

# useful ops for wrting cell wrapper
from tensorflow.python.ops import rnn_cell_impl


# weights / biases initializer
def weights(name, shape):
  return tf.get_variable(
    name, 
    shape=shape, 
    initializer=tf.contrib.layers.xavier_initializer()
    )


def biases(name, shape):
  return tf.get_variable(
    name, 
    initializer=tf.zeros(shape)
    )


class WritingCell(rnn_cell_impl.RNNCell):
  """ First three layers of synthesis network introduced by
      https://arxiv.org/pdf/1308.0850.pdf

      Intend to provide a wrapped cell to be called by Tensorflow
      interfaces such as dynamic_rnn.
  """
  def __init__(self, batch_size, char_vec_dim, max_char_len, num_mixture, dim_rec):
    def expand_and_repeat(data, axis, num_repeat):
      data = np.expand_dims(data, axis=axis)
      data = np.repeat(data, repeats=num_repeat, axis=axis)
      return data
    # super(WritingCell, self).__init__(_reuse=True)
    self.num_mixture = num_mixture
    self.dim_rec = dim_rec
    self._cell_0 = tf.contrib.rnn.BasicLSTMCell(dim_rec)
    self._cell_1 = tf.contrib.rnn.BasicLSTMCell(dim_rec)

    self.window_dim = char_vec_dim
    self.batch_size = batch_size
    self.num_mixture = num_mixture
    self.max_char_len = max_char_len

    # char_index [batch, num_mixture, max_char_len]
    char_index = np.arange(max_char_len, dtype=np.float32)
    char_index = expand_and_repeat(char_index, 0, num_mixture)
    char_index = expand_and_repeat(char_index, 0, batch_size)
    self.char_index = tf.constant(char_index)

    dim_coeff = self.num_mixture * 3
    self.W = weights("window_W", [self.dim_rec, dim_coeff])
    self.b = biases("window_b", [dim_coeff])

  @property
  def state_size(self):
    size = (self._cell_0.state_size, # first cell state 
            self._cell_1.state_size, # second cell state
            (self.batch_size, self.num_mixture, 1), # kappa
            (self.batch_size, self.window_dim), # window vector
            (self.batch_size, self.max_char_len, self.window_dim)) # char_vec
    return size

  @property
  def output_size(self):
    size = self._cell_1.output_size
    return size

  def __call__(self, cell_input, state):
    """ inputs :: [batch, cell_dim] 
    """
    # char_vec [batch, max_char_len, window_dim]
    cell_state_0, cell_state_1, last_kappa, last_window_vec, last_phi, char_vec = state

    def coeff_mapping(cell_output, last_kappa):
      """ Eq. 48-51
      """
      # output mapping
      outputs = tf.matmul(cell_output, self.W) + self.b

      # split the coeff.
      z_alpha, z_beta, z_kappa = tf.split(
        axis=1, num_or_size_splits=3, value=outputs)

      # output functions
      alpha = tf.expand_dims(tf.exp(z_alpha), axis=2) # [batch, num_mixture, 1]
      beta = tf.expand_dims(tf.exp(z_beta), axis=2) # [batch, num_mixture, 1]
      kappa = last_kappa + tf.expand_dims(tf.exp(z_kappa), axis=2) # [batch, num_mixture, 1]

      return alpha, beta, kappa
    
    # first recurrent layer
    cell_input_0 = tf.concat([cell_input, last_window_vec], 1)
    with tf.variable_scope("rnn_0"):
      cell_output_0, new_state_0 = self._cell_0(cell_input_0, cell_state_0)

    alpha, beta, kappa = coeff_mapping(cell_output_0, last_kappa)

    # Eq. 46
    # dims of time t:
    # char_index [batch, num_mixture, max_char_len]
    # kappa, beta, alpha [batch, num_mixture, 1]
    # phi [batch, max_char_len, 1]
    phi = alpha * tf.exp(-beta * tf.square(-self.char_index + kappa)) 
    phi = tf.transpose(tf.reduce_sum(phi, axis=1, keep_dims=True), [0, 2, 1])

    # Eq. 47
    # dims of time t:
    # window_vec [batch, window_dim]
    # phi [batch, max_char_len, 1]
    # char_vec [batch, max_char_len, window_dim]
    window_vec = tf.reduce_sum(char_vec * phi, 1)

    # second recurrent layer
    cell_input_1 = tf.concat([cell_input, cell_output_0, window_vec], 1)
    with tf.variable_scope("rnn_1"):
      cell_output_1, new_state_1 = self._cell_1(cell_input_1, cell_state_1)

    new_state = (new_state_0, new_state_1, kappa, window_vec, phi, char_vec)

    return cell_output_1, new_state

  def zero_state(self, char_vec):
    state_0 = self._cell_0.zero_state(self.batch_size, tf.float32)
    state_1 = self._cell_1.zero_state(self.batch_size, tf.float32)
    # intialize the kappa param in eq. 51 as 0
    init_kappa = tf.zeros([self.batch_size, self.num_mixture, 1]) 
    init_window_vec = tf.zeros((self.batch_size, self.window_dim))
    # place a dummy phi, for later extracting
    phi = tf.zeros([self.batch_size, self.max_char_len, 1])
    state = (state_0, state_1, init_kappa, init_window_vec, phi, char_vec)
    return state


class Model():
  def __init__(self, args, sampling=False, bias=1.):

    self.board_writer = None

    def coeff_mapping(cell_outputs, num_mixture=20, bias=1.):

      batch_size, _, dim_rec = cell_outputs.shape
      cell_outputs_flat = tf.reshape(cell_outputs, [-1, dim_rec.value]) # [batch*time, dim_rec]

      # 6 = 2 of means, 2 of variance, 1 of cluster prior, 1 of correlation
      # 1 = end of stroke probability
      dim_coeff = 6 * num_mixture + 1

      # output mapping
      W = weights("out_W", [dim_rec, dim_coeff])
      b = biases("out_b", [dim_coeff])
      outputs = tf.matmul(cell_outputs_flat, W) + b

      # split the coeff. along different clusters
      z_eos = outputs[:, 0:1]
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
        axis=1, num_or_size_splits=6, value=outputs[:, 1:]) 

      # output functions
      eos = tf.sigmoid(z_eos)
            
      pi = tf.nn.softmax(z_pi * (1+bias))

      mu1 = z_mu1
      mu2 = z_mu2

      sigma1 = tf.exp(z_sigma1 - bias)
      sigma2 = tf.exp(z_sigma2 - bias)

      corr = tf.tanh(z_corr)

      return eos, pi, mu1, mu2, sigma1, sigma2, corr

    def loss_func(evidence, eos, pi, mu1, mu2, sigma1, sigma2, corr, weights):
      # hard code the output dim as 3
      evidence_flat = tf.reshape(evidence, [-1, 3])

      def single_mixture_loss(x1, x2, mu1, mu2, s1, s2, corr):
        diff1 = x1 - mu1
        diff2 = x2 - mu2
        s1s2 = s1 * s2
        corr2 = tf.square(corr)
        norm_exponent = - tf.square( diff1 / s1) \
                        - tf.square( diff2 / s2) \
                        + (2. * corr * diff1 * diff2 ) / s1s2
        score = norm_exponent / 2 / (1 - corr2)
        normalizer = 1. / (2 * math.pi * s1s2 * tf.sqrt(1 - corr2))
        result = normalizer * tf.exp(score)
        return result

      def full_loss(x1, x2, eos_evid, eos, pi, mu1, mu2, s1, s2, corr):
        sml = single_mixture_loss(x1, x2, mu1, mu2, s1, s2, corr)
        coord_loss = tf.reduce_sum(pi * sml, 1, keep_dims=True) # [batch*time, 1]
        stroke_loss = eos_evid * eos + (1 - eos_evid) * (1 - eos) # [batch*time, 1]
        coord_loss = tf.maximum(coord_loss, 1e-20)
        stroke_loss = tf.maximum(stroke_loss, 1e-20)
        log_loss = - tf.log(coord_loss) - tf.log(stroke_loss) # [batch*time, 1]
        return log_loss

      x1 = evidence_flat[:, 0:1]
      x2 = evidence_flat[:, 1:2]
      eos_evid = evidence_flat[:, 2:3]

      loss = full_loss(x1, x2, eos_evid, eos, pi, mu1, mu2, sigma1, sigma2, corr)

      loss = tf.reshape(loss, [self.args.batch_size, -1, 1])

      loss = loss * weights

      loss_per_step = tf.reduce_sum(loss) / tf.reduce_sum(weights)

      return loss_per_step

    self.args = args

    if sampling:
      args.batch_size = 1
    # placeholders
    # the dimension looks like this: [batch, time, data_dimension]
    self.inputs = inputs = tf.placeholder(
      tf.float32, 
      [args.batch_size, None, 3]
      )
    self.targets = targets = tf.placeholder(
      tf.float32,
      [args.batch_size, None, 3]
      )
    self.weights = tf.placeholder(
      tf.float32, 
      [args.batch_size, None, 1]
      )
    cell_inputs = inputs

    # cells
    def cell_init(dim_rec):
      return tf.contrib.rnn.BasicLSTMCell(dim_rec)

    if args.mode == "prediction":

      cell = tf.contrib.rnn.MultiRNNCell(
        [cell_init(args.dim_rec) for _ in range(args.num_layers)]
        )

      self.state_in = cell.zero_state(args.batch_size, dtype=tf.float32)

    elif args.mode == "synthesis":

      if args.num_layers < 3:
        raise ValueError("The synthesis net should have num_layers >= 3!")

      self.texts = tf.placeholder(
        tf.float32,
        [args.batch_size, None, args.char_vec_dim]
        )
      cell = WritingCell(args.batch_size, args.char_vec_dim, 
        self.args.max_char_len, num_mixture=20, dim_rec=args.dim_rec)
      self.state_in = cell.zero_state(self.texts)

      if args.num_layers > 3:
        cell_list = [cell_init(args.dim_rec) for _ in range(args.num_layers - 3)]
        self.state_in = [self.state_in] \
          + [_c.zero_state(args.batch_size, dtype=tf.float32) for _c in cell_list]
        self.state_in = tuple(self.state_in)
        cell = tf.contrib.rnn.MultiRNNCell([cell] + cell_list)

    else:
      raise NotImplementedError("Unknown mode %s" % args.mode)

    # build structures
    cell_outputs, self.state_out = tf.nn.dynamic_rnn(
      cell=cell,
      inputs=cell_inputs,
      initial_state=self.state_in
      )

    eos, pi, mu1, mu2, sigma1, sigma2, corr = coeff_mapping(cell_outputs, bias=bias)

    self.eos = eos
    self.pi = pi
    # stack the mu and cov matrix, dim mu = [batch*time, cluster, input]
    self.mu = tf.stack([mu1, mu2], 2)
    # dim cov = [batch*time, cluster, rows, cols]
    col1 = tf.stack([sigma1*sigma1, corr*sigma1*sigma2], 2)
    col2 = tf.stack([corr*sigma1*sigma2, sigma2*sigma2], 2)
    self.Sigma = tf.stack([col1, col2], 3)
    self.corr = corr

    # loss: customized loss
    self.loss = loss = loss_func(targets, eos, pi, mu1, mu2, 
      sigma1, sigma2, corr, self.weights)

    # optimizer
    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
      tf.gradients(loss, tvars),
      args.max_grad_norm
      )
    optimizer = tf.train.AdamOptimizer(self.lr)
    # optimizer = tf.train.RMSPropOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        

  def sample(self, sess, sample_len, 
      initial_point=[[[0., 0., 1.]]], texts=None, ref_texts=None):

    assert sample_len >= 1
    if ref_texts is not None:
      texts = np.concatenate((ref_texts, texts), axis=1)

    def sample_2d_gaussian(mu, Sigma):
      x1, x2 = np.random.multivariate_normal(mu, Sigma+1e-20*np.eye(2), 1)[0]
      return x1, x2

    def sample_cluster(pi):
      rand = random.random()
      accumulate = 0.
      for i in range(len(pi)):
        accumulate += pi[i]
        if accumulate >= rand:
          return i
      raise ValueError("Cannot sample a cluster!")

    def sample_mix_gaussian(pi, mu, Sigma):
      cluster_idx = sample_cluster(pi)
      x1, x2 = sample_2d_gaussian(mu[cluster_idx], Sigma[cluster_idx])
      return x1, x2

    def sample_stroke_point(eos, pi, mu, Sigma):
      x1, x2 = sample_mix_gaussian(pi, mu, Sigma)
      if random.random() > eos:
        return np.asarray([x1, x2, 0.])
      return np.asarray([x1, x2, 1.])

    def checkstop(states):
      phi = states[-2] # the window vector of size [1, char_len+1, 1]
      if phi[0, -1, 0] / phi[0, :-1, 0].max() > 10: # TODO: 10 is a hacky number
        return True
      return False

    init_feed = {self.inputs: initial_point}
    if self.args.mode == "synthesis" and texts is not None:
      init_feed[self.texts] = texts
    # initial run
    outputs = sess.run(
      [self.eos, self.pi, self.mu, self.Sigma] + nest.flatten(self.state_out),
      feed_dict=init_feed
      )

    eos, pi, mu, Sigma = outputs[:4]
    
    prev_state = outputs[4:]

    x = sample_stroke_point(eos[-1], pi[-1], mu[-1], Sigma[-1])

    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    prev_x[0][0] = x
    
    strokes = []
    window_vec_list = []

    for i in range(sample_len):
      feed = {
        tensor: s 
        for tensor, s in zip(nest.flatten(self.state_in), prev_state)
        }
      feed[self.inputs] = prev_x

      outputs = sess.run(
        [self.eos, self.pi, self.mu, self.Sigma]+nest.flatten(self.state_out),
        feed_dict=feed
        )

      eos, pi, mu, Sigma = outputs[:4]
      state = outputs[4:]

      if checkstop(state):
        break

      x = sample_stroke_point(eos[0], pi[0], mu[0], Sigma[0])

      strokes.append(np.squeeze(x))

      prev_x = np.zeros((1, 1, 3), dtype=np.float32)
      prev_x[0][0] = x
      
      prev_state = state

      window_vec_list.append(np.squeeze(state[-2])) # the window vectors

    strokes = np.asarray(strokes)
    window_vecs = np.asarray(window_vec_list)
    if ref_texts is not None:
      window_vecs = window_vecs[:, ref_texts.shape[1]:]

    return strokes, window_vecs


  def train(self, sess, sequence, targets, weights, 
            texts, subseq_length, step_count):
    """ Cut the training sequences into multiple sub-sequences for training
    """

    if self.board_writer is None:
      # tensorboard
      tf.summary.scalar('MoG_loss', self.loss)
      self.merged = tf.summary.merge_all()
      self.board_writer = tf.summary.FileWriter(
        self.args.summary_dir + "/" + str(int(time.time())), 
        sess.graph
        )

    loss_list = []
    init_feed = {}
    init_feed[self.inputs] = sequence[:, :subseq_length, :]
    init_feed[self.targets] = targets[:, :subseq_length, :]
    init_feed[self.weights] = weights[:, :subseq_length, :]
    if self.args.mode == "synthesis":
      init_feed[self.texts] = texts 
    outputs = sess.run(
      [self.loss, self.merged, self.train_op] + nest.flatten(self.state_out),
      init_feed
      )
    train_loss, summary, _ = outputs[:3]
    prev_state = outputs[3:]
    self.board_writer.add_summary(summary, step_count)
    step_count += 1

    loss_list.append(train_loss)
    
    while sequence.shape[1] > subseq_length:
      sequence = sequence[:, subseq_length:, :]
      targets = targets[:, subseq_length:, :]
      weights = weights[:, subseq_length:, :]

      feed = {
        tensor: s 
        for tensor, s in zip(nest.flatten(self.state_in), prev_state)
        }
      feed[self.inputs] = sequence[:, :subseq_length, :]
      feed[self.targets] = targets[:, :subseq_length, :]
      feed[self.weights] = weights[:, :subseq_length, :]

      outputs = sess.run(
        [self.loss, self.merged, self.train_op]+nest.flatten(self.state_out),
        feed_dict=feed
        )
      train_loss, summary, _ = outputs[:3]
      prev_state = outputs[3:]

      self.board_writer.add_summary(summary, step_count)
      step_count += 1

      loss_list.append(train_loss)

    return loss_list, step_count