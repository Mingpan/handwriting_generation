import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import svgwrite
from IPython.display import SVG, display


VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
REV_VOCAB = {char: digit for digit, char in enumerate(VOCAB)}


def rev_one_hot(data):
  """ Convert 2D one hot vector back
  """
  vec = []
  for i in range(data.shape[0]):
    new_int = np.where(data[i]==1)[0][0]
    if new_int < len(VOCAB):
      new_char = VOCAB[new_int]
      vec.append(new_char)
    else:
      vec.append(str(new_int))
  return vec


def one_hot_vectorize(data, max_dim):
  """ Convert 1D data into 2D one hot vector
  """
  onehot = np.zeros((len(data), max_dim))
  onehot[np.arange(len(data)), data] = 1.
  return onehot


def texts_prep_for_sampling(texts):
    ints = []
    for i, char in enumerate(texts):
      if char in REV_VOCAB:
        ints.append(REV_VOCAB[char])
      else:
        ints.append(len(VOCAB)) # out of vocab token

    text_vec = one_hot_vectorize(ints, len(VOCAB)+1)
    # num_fill = 1
    # chars_to_fill = np.zeros((num_fill, self.char_vec_dim))
    # chars_to_fill[:, REV_VOCAB[" "]] = 1.
    # text_vec = np.concatenate((text_vec, chars_to_fill), axis=0)
    return np.expand_dims(text_vec, axis=0)

def numpy_fillzeros(data):
    """ Reshape a list of 2D arrays to a 3D array.
        Insufficient length arrays with be filled with 0s.
        test input:
            [np.array([[1., 1., 1.]]), 
             np.array([[2., 2., 0.], [3., 3., 1.]])]
        expected output:
            np.array([[[1., 1., 1.],
                      [0., 0., 0.]],
                     [[2., 2., 0.],
                      [3., 3., 1.]]])
            np.array([[[1.],
                      [0.]],
                     [[1.],
                      [1.]]])
    """
    # get the lengths of all arrays
    lens = np.array([ d.shape[0] for d in data]) # [batch]
    # mask out the insufficient part
    mask2d = np.arange(lens.max()) < lens[:, None] # [batch, max_len]
    mask = np.repeat(mask2d[:, :, None], 3, axis=2) # [batch, max_len, 3]
    # setup a matrix of the right size and fill in the values
    out = np.zeros(mask.shape) # [batch, max_len, 3]
    out[mask] = np.concatenate(data).flatten()
    return out, mask2d[:,:,None].astype(np.float32)


def get_bounds(data, factor):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
    
  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
    
  return (min_x, max_x, min_y, max_y)

# old version, where each path is entire stroke (smaller svg size, but have to keep same color)
def draw_strokes(data, factor=10, svg_filename = 'sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
    
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  lift_pen = 1
    
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
    
  command = "m"

  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "

  the_color = "black"
  stroke_width = 1

  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))

  dwg.save()
  display(SVG(dwg.tostring()))

def draw_strokes_eos_weighted(stroke, param, factor=10, svg_filename = 'sample_eos.svg'):
  c_data_eos = np.zeros((len(stroke), 3))
  for i in range(len(param)):
    c_data_eos[i, :] = (1-param[i][6][0])*225 # make color gray scale, darker = more likely to eos
  draw_strokes_custom_color(stroke, factor = factor, svg_filename = svg_filename, color_data = c_data_eos, stroke_width = 3)

def draw_strokes_random_color(stroke, factor=10, svg_filename = 'sample_random_color.svg', per_stroke_mode = True):
  c_data = np.array(np.random.rand(len(stroke), 3)*240, dtype=np.uint8)
  if per_stroke_mode:
    switch_color = False
    for i in range(len(stroke)):
      if switch_color == False and i > 0:
        c_data[i] = c_data[i-1]
      if stroke[i, 2] < 1: # same strike
        switch_color = False
      else:
        switch_color = True
  draw_strokes_custom_color(stroke, factor = factor, svg_filename = svg_filename, color_data = c_data, stroke_width = 2)

def draw_strokes_custom_color(data, factor=10, svg_filename = 'test.svg', color_data = None, stroke_width = 1):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
    
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  for i in range(len(data)):

    x = float(data[i,0])/factor
    y = float(data[i,1])/factor

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y

    if (lift_pen == 1):
      p = "M "+str(abs_x)+","+str(abs_y)+" "
    else:
      p = "M +"+str(prev_x)+","+str(prev_y)+" L "+str(abs_x)+","+str(abs_y)+" "

    lift_pen = data[i, 2]

    the_color = "black"

    if (color_data is not None):
      the_color = "rgb("+str(int(color_data[i, 0]))+","+str(int(color_data[i, 1]))+","+str(int(color_data[i, 2]))+")"

    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill(the_color))
  dwg.save()
  display(SVG(dwg.tostring()))

def draw_strokes_pdf(data, param, factor=10, svg_filename = 'sample_pdf.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)

  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  num_mixture = len(param[0][0])

  for i in range(len(data)):

    x = float(data[i,0])/factor
    y = float(data[i,1])/factor

    for k in range(num_mixture):
      pi = param[i][0][k]
      if pi > 0.01: # optimisation, ignore pi's less than 1% chance
        mu1 = param[i][1][k]
        mu2 = param[i][2][k]
        s1 = param[i][3][k]
        s2 = param[i][4][k]
        sigma = np.sqrt(s1*s2)
        dwg.add(dwg.circle(center=(abs_x+mu1*factor, abs_y+mu2*factor), r=int(sigma*factor)).fill('red', opacity=pi/(sigma*sigma*factor)))

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y

  dwg.save()
  display(SVG(dwg.tostring()))



class DataLoader():
  def __init__(self, batch_size=50, scale_factor=10, subseq_length=100, limit=500):
    self.subseq_length = subseq_length
    self.data_dir = "./data"
    self.batch_size = batch_size
    # self.seq_length = seq_length
    self.scale_factor = scale_factor # divide data by this factor
    self.limit = limit # removes large noisy gaps in the data

    data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
    raw_data_dir = self.data_dir+"/lineStrokes"
    text_dir = self.data_dir+"/ascii"

    if not (os.path.exists(data_file)) :
        print("creating training data pkl file from raw source")
        self.preprocess(raw_data_dir, text_dir, data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def preprocess(self, data_dir, text_dir, data_file):
    # create data file from raw xml files from iam handwriting source.

    # build the list of xml files
    filelist = []
    # Set the directory you want to start from
    rootDir = data_dir
    for dirName, subdirList, fileList in os.walk(rootDir):
      #print('Found directory: %s' % dirName)
      for fname in fileList:
        #print('\t%s' % fname)
        filelist.append(dirName+"/"+fname)

    # function to read each individual xml file
    def getStrokes(filename):
      tree = ET.parse(filename)
      root = tree.getroot()

      result = []

      x_offset = 1e20
      y_offset = 1e20
      y_height = 0
      for i in range(1, 4):
        x_offset = min(x_offset, float(root[0][i].attrib['x']))
        y_offset = min(y_offset, float(root[0][i].attrib['y']))
        y_height = max(y_height, float(root[0][i].attrib['y']))
      y_height -= y_offset
      x_offset -= 100
      y_offset -= 100

      for stroke in root[1].findall('Stroke'):
        points = []
        for point in stroke.findall('Point'):
          points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
        result.append(points)

      return result

    # converts a list of arrays into a 2d numpy int16 array
    def convert_stroke_to_array(stroke):

      n_point = 0
      for i in range(len(stroke)):
        n_point += len(stroke[i])
      stroke_data = np.zeros((n_point, 3), dtype=np.int16)

      prev_x = 0
      prev_y = 0
      counter = 0

      for j in range(len(stroke)):
        for k in range(len(stroke[j])):
          stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
          stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
          prev_x = int(stroke[j][k][0])
          prev_y = int(stroke[j][k][1])
          stroke_data[counter, 2] = 0
          if (k == (len(stroke[j])-1)): # end of stroke
            stroke_data[counter, 2] = 1
          counter += 1
      return stroke_data

    def get_ascii(stroke_filename):
        """get the ascii data, inferred from the stroke filename
        """
        # divide the stroke_filename into:
        # real_filename + (-) +  line_index + (.xml)
        filename, _ = stroke_filename.rsplit(".", 1)
        real_filename, line_index = filename.rsplit("-", 1)

        # get the ascii file path
        ascii_filename = real_filename.replace(data_dir, text_dir) + ".txt"
        with open(ascii_filename, "r") as f:
            texts = f.readlines()

        for i in range(len(texts)):
            if texts[i].strip() == "CSR:":
                try:
                    # skip an empty line, then add the line_index
                    return texts[i + 1 + int(line_index)].strip()
                except:
                    print("Cannot find the characters for stroke file %s"
                          % real_filename)
                    return None

    def char2int(chars, vocab):
      """ This func. reads all the chars
      """
      ints = []
      for char in chars:
        if char not in vocab:
          ints.append(len(vocab))
          vocab.append(char)
        else:
          ints.append(vocab.index(char))
      return ints

    def char2int_reduced(chars):
      """ This func. has a predefined vocab.
      """
      out_of_vocab = len(VOCAB)
      ints = []
      for char in chars:
        if char in REV_VOCAB:
          ints.append(REV_VOCAB[char])
        else:
          ints.append(out_of_vocab)
      return ints


    # build stroke database of every xml file inside iam database
    strokes = []
    text = []
    vocab = [] # vocab[int] = char
    for i in range(len(filelist)):
      if (filelist[i][-3:] == 'xml'):
        chars = get_ascii(filelist[i])
        if chars:
            # if the characters can be found
            print('processing '+filelist[i])
            strokes.append(convert_stroke_to_array(getStrokes(filelist[i])))
            # text.append(char2int(chars, vocab))
            text.append(char2int_reduced(chars))

    rev_vocab = {} # rev_vocab[char] = int
    for i, char in enumerate(vocab):
        rev_vocab[char] = i

    preprocessed = {}
    preprocessed["strokes"] = strokes
    preprocessed["text"] = text
    # preprocessed["int2char"] = vocab
    # preprocessed["char2int"] = rev_vocab
    preprocessed["int2char_53"] = VOCAB
    preprocessed["char2int_53"] = REV_VOCAB

    with open(data_file,"wb") as f:
        pickle.dump(preprocessed, f, protocol=2)

  def load_preprocessed(self, data_file):
    with open(data_file,"rb") as f:
        self.raw_data = pickle.load(f)

    strokes = self.raw_data["strokes"]
    # self.rev_vocab = self.raw_data["char2int"]
    # self.vocab = self.raw_data["int2char"]

    self.rev_vocab = self.raw_data["char2int_53"]
    self.vocab = self.raw_data["int2char_53"]

    texts = self.raw_data["text"]
    self.max_char_len = max(len(t) for t in texts)
    self.char_vec_dim = len(self.vocab) + 1 # dim of char vector=vocab+unknown token

    # goes thru the list
    self.data = {"strokes": [], "texts": []} 
    self.valid_data = {"strokes": [], "texts": []}
    counter = 0

    # cur_data_counter = 0
    for stroke, text in zip(strokes, texts):
      # removes large gaps from the stroke
      stroke = np.minimum(stroke, self.limit)
      stroke = np.maximum(stroke, -self.limit)
      stroke = np.array(stroke,dtype=np.float32)
      stroke[:,0:2] /= self.scale_factor
      # cur_data_counter = cur_data_counter + 1

      text_vec = one_hot_vectorize(text, self.char_vec_dim)

      # fill text_vec to longest char len
      num_fill = self.max_char_len - text_vec.shape[0]
      chars_to_fill = np.zeros((num_fill, self.char_vec_dim))
      chars_to_fill[:, REV_VOCAB[" "]] = 1.
      text_vec = np.concatenate((text_vec, chars_to_fill), axis=0)

      # validation data is not accurate for synthesis task
      # if cur_data_counter % 200 == 0:
      #   self.valid_data["strokes"].append(stroke)
      #   self.valid_data["texts"].append(text_vec)
      # else:
      #   self.data["strokes"].append(stroke)
      #   self.data["texts"].append(text_vec)
      #   counter += int(len(stroke)/((self.mean_length+2)))

      self.data["strokes"].append(stroke)
      self.data["texts"].append(text_vec)
      counter += int(np.ceil(len(stroke)/(self.subseq_length+1)))

    print("train data: {}, valid data: {}"
          .format(len(self.data["strokes"]), len(self.valid_data["strokes"])))
    self.num_batches = int(np.ceil(counter / self.batch_size))
    self.num_sequences = len(self.data["strokes"])

  # validation data is not needed
  # def validation_data(self):
  #   # returns validation data
  #   x_batch = []
  #   y_batch = []
  #   c_batch = []
  #   for i in range(self.batch_size):
  #     stroke = self.valid_data["strokes"][i%len(self.valid_data["strokes"])]
  #     text_vec = self.valid_data["texts"][i%len(self.valid_data["texts"])]
  #     max_len = stroke.shape[0]
  #     text_len = text_vec.shape[0]
  #     x_batch.append(np.copy(stroke[:max_len-1]))
  #     y_batch.append(np.copy(stroke[1:max_len]))
  #     c_batch.append(np.copy(text_vec))
  #   x_batch, x_weights = numpy_fillzeros(x_batch)
  #   y_batch, _ = numpy_fillzeros(y_batch)
  #   return x_batch, y_batch, x_weights, np.asarray(c_batch)

  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    c_batch = []
    for i in range(self.batch_size):
      stroke = self.data["strokes"][self.pointer]
      text = self.data["texts"][self.pointer]
      # n_batch = int(len(stroke)/((self.seq_length+2))) # number of equiv batches this datapoint is worth
      # idx = random.randint(0, len(stroke)-self.seq_length-2)
      # x_batch.append(np.copy(str1.0/float(n_batch)): # adjust sampling probability.
      #   #if this is a long datapoint, sample this data more with higher probability
      #   self.tick_batch_pointer()
      
      # using entire sequence, because we don't know where characters starts/ends
      max_len = stroke.shape[0]
      text_len = text.shape[0]
      x_batch.append(np.copy(stroke[:max_len-1]))
      y_batch.append(np.copy(stroke[1:max_len]))
      c_batch.append(np.copy(text)) # TODO hard code the length of chars if needed
      self.tick_batch_pointer()
    x_batch, x_weights = numpy_fillzeros(x_batch)
    y_batch, _ = numpy_fillzeros(y_batch)
    return x_batch, y_batch, x_weights, np.asarray(c_batch)

  def load_sample(self, idx):
    """ Load a sample batch by index. For copying the style.
    """
    idx = idx % self.num_sequences
    stroke = self.data["strokes"][idx]
    text = self.data["texts"][idx]
    return np.expand_dims(stroke, 0), np.expand_dims(text, 0)

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data["strokes"])):
      self.pointer = 0
  def reset_batch_pointer(self):
    self.pointer = 0

