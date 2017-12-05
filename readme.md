
<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/this_is_a_handwriting_generation_model_13.svg" width="1000" height="150">

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/it_is_able_to_write_texts_as_given_13.svg" width="1000" height="150">

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/or_change_writing_style_2.svg" width="700" height="150">

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/if_asked_to_554.svg" width="400" height="150">

# Dependencies
python3.5,  
[tensorflow](https://www.tensorflow.org/install/) r1.4 or r1.2,  
svgwrite (installable with pip),  
ipython (installable with pip),  

Necessary for training / copying writing style from training set:  
[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database) the ascii dataset `ascii-all.tar.gz` and xml dataset `data/lineStrokes-all.tar.gz` Extract them in the `data/` dir.

# Resources
## Implementation based on the paper
[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
## Dataset provided by
[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database)
## Used code from the following repo
https://github.com/hardmaru/write-rnn-tensorflow  
Based on this, I built the synthesis net, enable it to generate characters as specified. To mimic a specific handwriting in training set is also possible.  
I also got some useful inspirations from the following repo  
https://github.com/snowkylin/rnn-handwriting-generation

# Usage
## Mode
Two modes available, `--mode prediction` or `--mode synthesis`, for freely generating or generating with character supervision.
## Training
`python train.py`, and try `python train.py -h` for possible input arguments.
## Generating (Sampling)
`python sample.py`, and try `--texts "<the texts you want to write>"` when given a synthesis model.  
`--model_dir <your_model_dir>` specifies which model should be loaded.  
`--bias` is a non-negative float that specifies how risky the model will be during generating, e.g. `--bias 0` for as random as possible.  
`--copy_style` for specifying a sample in training set to copy style from, e.g. `--copy_style 2`.

# Example
## Naive generating given texts
`python sample.py` the result would be a handwriting as follows.  

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/sample.normal.svg" width="500" height="100">  

And a window alignment figure will be generated, which tells the connection between characters (vertical axis) and respective strokes (horizontal axis). This should reveal if the generation is performing well.  

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/sample_window.svg">

## Generating given texts with a style in the training set
(For this you need to download the training dataset first, see dependencies.)  
`python sample.py --copy_style 2` the result would be a handwriting that mimics the second training example.   

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/sample_copy.normal.svg" width="500" height="100">    

Similarly, a window alignment figure will be generated.  

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/sample_copy_window.svg">    

A reference handwriting will be drawn. One can therefore tell if the copying was good enough.  

<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/sample_copy.normal_ref.svg" width="500" height="100">    

# License for the code
MIT

# Trouble shooting
## OOM Error
It could be that the allocated tensors are too big, try a smaller batch size etc.  
## Need better handwritings!
Copying only from good training examples can increase the quality a bit. Also, I didn't try a lot of hyperparameter settings. Train a model of your own using better hyperparameters if you like.


<img src="https://github.com/Mingpan/handwriting_generation/blob/master/samples/any_feedback_is_welcome.svg" width="700" height="150">
