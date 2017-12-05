""" The main script for module RNN
"""
import os
import time
import math
import argparse
import pickle

import numpy as np
import tensorflow as tf
# from tensorflow.contrib.framework import nest

from utils import DataLoader, draw_strokes
from model import Model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dim_rec', type=int, default=400,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=3,
                     help='number of layers in the RNN. ' \
                     'Needs to be larger than 3 for synthesis.')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--num_epochs', type=int, default=200,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=5,
                     help='save frequency by epoches')
  parser.add_argument('--model_dir', type=str, default='checkpoints',
                     help='directory to save model to')
  parser.add_argument('--summary_dir', type=str, default='summary',
                     help='directory to save tensorboard info')
  parser.add_argument('--max_grad_norm', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=1.0,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--data_scale', type=float, default=20,
                     help='factor to scale raw data down by')
  parser.add_argument('--mode', type=str, default='synthesis',
                     help='prediction / synthesis' )
  parser.add_argument('--load_model', type=str, default=None,
                     help='Reload a model checkpoint and restore training.' )
  parser.add_argument('--bptt_length', type=int, default=300,
                     help='How many steps should the gradients pass back.' )
  
  args = parser.parse_args()

  train(args)

def train(args):
  data_loader = DataLoader(args.batch_size, args.data_scale, args.bptt_length)
  data_loader.reset_batch_pointer()

  # model needs to know the dim of one-hot vectors
  args.char_vec_dim = data_loader.char_vec_dim
  # also the max length of char vector
  args.max_char_len = data_loader.max_char_len

  if args.model_dir != '' and not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

  with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(args, f)
  print("hyperparam. saved.")

  model = Model(args)

  # training
  with tf.Session() as sess:

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    if args.load_model is not None:
        saver.restore(sess, args.load_model)
        _, ep_start = args.load_model.rsplit("-", 1)
        ep_start = int(ep_start)
        model_steps = int(ep_start * data_loader.num_batches)
    else:
        ep_start = 0
        model_steps = last_model_steps = 0

    last_time = time.time()

    for ep in range(ep_start, args.num_epochs):
      ep_loss = []
      sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** ep)))

      for i in range(int(data_loader.num_sequences / args.batch_size)):
        idx = ep * data_loader.num_sequences + i * args.batch_size
        start = time.time()
        x, y, w, c = data_loader.next_batch()

        loss_list, model_steps = model.train(
          sess=sess, 
          sequence=x, 
          targets=y, 
          weights=w, 
          texts=c, 
          subseq_length=args.bptt_length, 
          step_count=model_steps
          )

        ep_loss += loss_list

        if model_steps - last_model_steps >= 100:
          last_model_steps = model_steps
          new_time = time.time()
          print(
            "Sequence %d/%d (epoch %d), batch %d, train_loss = %.3f, time/(100*batch) = %.3f" 
            % (
                idx,
                args.num_epochs * data_loader.num_sequences,
                ep,
                model_steps,
                np.mean(loss_list),
                new_time - last_time
              ),
            flush=True
            )
          last_time = new_time
      print("Epoch %d completed, average train loss %.6f" % (ep, np.mean(ep_loss)))
      if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
      if (ep+1) % args.save_every == 0:
        checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
        saver.save(sess, save_path=checkpoint_path, global_step = (ep+1))
        print("model saved.")


if __name__ == "__main__":
  main()
