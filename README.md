## This is a code base written for one of my PhD projects - the idea was to do the following: 

### 1) take some monophonic recordings 
### 2) do some pitch-and-onset analysis on them
### 3) convert the results to text
### 4) train a recurrent neural network on the text data 
### 5) generate new streams of melodic information from the trained model 
### 6) notate the results
### 7) make music based on the generated notation 

The code in this repo will get you as far as step 6 ;) 

The easiest way to run the program is to use the provided Colab notebook (click on the widget below). It'll pull in one of my datasets from HuggingFace, do all the data pre-processing, train a model and generate some notation. You just need a Google account and a HuggingFace account.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markhanslip/PhD_Ch6_Char_RNN/blob/main/Chapter_6_Notebook_Char_RNN_v2.ipynb)

To do: 
- Change the pitch analyser to CREPE. Currently I'm using Praat which works well on tenor saxophone (and of course voice) but doesn't generalise to lots of instruments.
- Grab the output of !apt-cache policy lilypond and add it as input to the lilypond markdown function in Generator.py to avoid future versioning issues (lilypond version is currently hard-coded). 

The neural network architecture used here is Char-RNN which was originally developed by https://github.com/karpathy/char-rnn and adapted from https://github.com/spro/char-rnn.pytorch

Pitch-based onset detection method is my own but Nick Collins had the idea first: https://composerprogrammer.com/research/pitchdetectforonset.pdf
