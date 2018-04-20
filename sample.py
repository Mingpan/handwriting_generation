import numpy as np
import tensorflow as tf

import time
import os
import pickle
import argparse

from utils import *
from model import Model
import random

import matplotlib.pyplot as plt
import svgwrite
from IPython.display import SVG, display

# main code (not in a main function since I want to run this script in IPython as well).

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                   help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=5000,
                   help='number of strokes to sample')
parser.add_argument('--scale_factor', type=int, default=1,
                   help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--model_dir', type=str, default='checkpoints',
                   help='directory to save model to')
parser.add_argument('--freeze_graph', dest='freeze_graph', action='store_true',
                   help='if true, freeze (replace variables with consts), prune (for inference) and save graph')
parser.add_argument('--texts', type=str, default='These are some sample texts',
                   help='texts to write, required for synthesis mode')
parser.add_argument('--bias', type=float, default=1.,
                   help='Positive float, indicates how wild the network '\
                        'should be during generating.')
parser.add_argument('--copy_style', type=int, default=None,
                   help='Copy the style from the training set.')

sample_args = parser.parse_args()

# add an empty char for specifying the start and ending
sample_args.texts = " " + sample_args.texts + " "

def erase_empty(c):
  # temp method to erase the empty chars
  l = c.shape[1]
  for i in range(l-1, -1, -1):
    if c[0, i, REV_VOCAB[" "]] == 1:
      c = c[:, :i, :]
    else:
      return c

with open(os.path.join(sample_args.model_dir, 'config.pkl'), 'rb') as f:
  saved_args = pickle.load(f)

if sample_args.copy_style is not None:
  data_loader = DataLoader(1, saved_args.data_scale)
  x, c = data_loader.load_sample(sample_args.copy_style)
  saved_args.max_char_len = len(sample_args.texts) + erase_empty(c).shape[1]
else:
  saved_args.max_char_len = len(sample_args.texts)
model = Model(saved_args, True, bias=sample_args.bias)
sess = tf.InteractiveSession()

saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(sample_args.model_dir)
print("loading model: ", ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke(texts=None):
  x_init = [[[0.,0.,1.]]]
  ref_texts = None
  if texts is not None:
    texts = texts_prep_for_sampling(texts)
    if sample_args.copy_style is not None:
      ref_texts = erase_empty(c)
      x_init = x
      draw_strokes(x[0], factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.normal_ref.svg')
    print("Printing the following text: ")
    print(''.join(rev_one_hot(texts[0])))
  strokes, window_vecs = model.sample(
    sess, 
    sample_args.sample_length, 
    initial_point=x_init, 
    texts=texts,
    ref_texts=ref_texts
    )
  
  draw_strokes(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.normal.svg')
  # draw_strokes_random_color(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.color.svg')
  # draw_strokes_random_color(strokes, factor=sample_args.scale_factor, per_stroke_mode = False, svg_filename = sample_args.filename+'.multi_color.svg')
  # draw_strokes_eos_weighted(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.eos_pdf.svg')
  # draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.pdf.svg')
  plot_window_vectors(window_vecs)
  return strokes

def plot_window_vectors(window_vecs):
  plt.imshow(np.squeeze(window_vecs).T, cmap='hot', aspect='auto')
  plt.savefig("window.svg")


def freeze_and_save_graph(sess, folder, out_nodes, as_text=False):
  ## save graph definition
  graph_raw = sess.graph_def
  graph_frz = tf.graph_util.convert_variables_to_constants(sess, graph_raw, out_nodes)
  ext = '.txt' if as_text else '.pb'
  #tf.train.write_graph(graph_raw, folder, 'graph_raw'+ext, as_text=as_text)
  tf.train.write_graph(graph_frz, folder, 'graph_frz'+ext, as_text=as_text)
    

if(sample_args.freeze_graph):
  freeze_and_save_graph(sess, sample_args.model_dir, ['data_out_mdn', 'data_out_eos', 'state_out'], False)

if saved_args.mode == "synthesis":
  strokes = sample_stroke(sample_args.texts)
else:
  strokes = sample_stroke()


