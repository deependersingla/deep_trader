import gzip
import os

import numpy as np
import six
from six.moves.urllib import request

from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np
import dateutil.parser
import pdb
import glob
import cPickle as pickle
import shelve
import six
import episodic_data
from six.moves.urllib import request

data = episodic_data.load_data("data.pkl",episode=10)
x_train, x_test = train_test_split(data, test_size=0.10, random_state=123)