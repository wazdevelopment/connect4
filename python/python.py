Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:05:16) [MSC v.1915 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk.classify.util, nltk.metrics\n",
    "from nltk.classify import NaiveBayesClassifier\n",
    "from nltk.corpus import movie_reviews\n",
    "import math"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "NLTK classifiers work with featstructs (http://www.nltk.org/_modules/nltk/featstruct.html). \n",
    "\n",
    "The NLTK implementation of Naive Bayes works as follows: In order to find the\n",
    "probability for a label, the algorithm first uses the Bayes rule to\n",
    "express P(label|features) in terms of P(label) and P(features|label):\n",
    "\n",
    "|                       P(label) * P(features|label)\n",
    "|  P(label|features) = ------------------------------\n",
    "|                              P(features)\n",
    "\n",
    "The algorithm then makes the 'naive' assumption that all features are\n",
    "independent, given the label:\n",
    "\n",
    "|                       P(label) * P(f1|label) * ... * P(fn|label)\n",
    "|  P(label|features) = --------------------------------------------\n",
    "|                                         P(features)\n",
    "\n",
    "Rather than computing P(features) explicitly, the algorithm just\n",
    "calculates the numerator for each label, and normalizes them so they\n",
    "sum to one:\n",
    "\n",
    "|                       P(label) * P(f1|label) * ... * P(fn|label)\n",
    "|  P(label|features) = --------------------------------------------\n",
    "|                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )\n",
    "\n",
    "For more information, please see here: http://www.nltk.org/_modules/nltk/classify/naivebayes.html "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here, we will construct a very simple dictionary which maps a feature name (word) to True if the word exists in the data. We will not use bag of words for this example because, for sentiment classification, whether a word occurs or not seems to matter more than its frequency. When it comes to Naive Bayes, this is called binary multinomial Naive Bayes. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def extract_word_feats(words):\n",
    "    return dict([(word, True) for word in words])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes is a generative classifier, i.e. it builds a model of each class and given an observation, it returns the class most likely to have generated the observation. \n",
    "\n",
    "The movie reviews corpus has 1000 positive files and 1000 negative files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "negids = movie_reviews.fileids('neg')\n",
    "posids = movie_reviews.fileids('pos')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to train the classifier, we initially need to creat feature-label pairs where the features will be a feature dictionary in the form of {word: True} and the label is either a \"pos\" or a \"neg\" label. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "negreviews = [(extract_word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]\n",
    "posreviews = [(extract_word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For instance, a feature-label pair could be: ({'\"': True, 'around': True,..., 'neg')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "({u'\"': True,\n",
       "  u\"'\": True,\n",
       "  u'(': True,\n",
       "  u')': True,\n",
       "  u',': True,\n",
       "  u'.': True,\n",
       "  u'a': True,\n",
       "  u'across': True,\n",
       "  u'acting': True,\n",
       "  u'action': True,\n",
       "  u'all': True,\n",
       "  u'and': True,\n",
       "  u'another': True,\n",
       "  u'are': True,\n",
       "  u'around': True,\n",
       "  u'average': True,\n",
       "  u'back': True,\n",
       "  u'baldwin': True,\n",
       "  u'bastard': True,\n",
       "  u'below': True,\n",
       "  u'big': True,\n",
       "  u'body': True,\n",
       "  u'brain': True,\n",
       "  u'bringing': True,\n",
       "  u'brother': True,\n",
       "  u'bug': True,\n",
       "  u'cgi': True,\n",
       "  u'chase': True,\n",
       "  u'comes': True,\n",
       "  u'course': True,\n",
       "  u'crew': True,\n",
       "  u'curtis': True,\n",
       "  u'damn': True,\n",
       "  u'deserted': True,\n",
       "  u'design': True,\n",
       "  u'do': True,\n",
       "  u'don': True,\n",
       "  u'donald': True,\n",
       "  u'drunkenly': True,\n",
       "  u'empty': True,\n",
       "  u'even': True,\n",
       "  u'feels': True,\n",
       "  u'few': True,\n",
       "  u'flash': True,\n",
       "  u'flashy': True,\n",
       "  u'for': True,\n",
       "  u'from': True,\n",
       "  u'get': True,\n",
       "  u'going': True,\n",
       "  u'good': True,\n",
       "  u'gore': True,\n",
       "  u'got': True,\n",
       "  u'h20': True,\n",
       "  u'halloween': True,\n",
       "  u'happy': True,\n",
       "  u'has': True,\n",
       "  u'he': True,\n",
       "  u'head': True,\n",
       "  u'her': True,\n",
       "  u'here': True,\n",
       "  u'hey': True,\n",
       "  u'hit': True,\n",
       "  u'if': True,\n",
       "  u'in': True,\n",
       "  u'into': True,\n",
       "  u'is': True,\n",
       "  u'it': True,\n",
       "  u'jamie': True,\n",
       "  u'just': True,\n",
       "  u'kick': True,\n",
       "  u'know': True,\n",
       "  u'lee': True,\n",
       "  u'let': True,\n",
       "  u'like': True,\n",
       "  u'likely': True,\n",
       "  u'likes': True,\n",
       "  u'little': True,\n",
       "  u'middle': True,\n",
       "  u'mir': True,\n",
       "  u'more': True,\n",
       "  u'movie': True,\n",
       "  u'much': True,\n",
       "  u'no': True,\n",
       "  u'nowhere': True,\n",
       "  u'occasional': True,\n",
       "  u'of': True,\n",
       "  u'on': True,\n",
       "  u'origin': True,\n",
       "  u'otherwise': True,\n",
       "  u'out': True,\n",
       "  u'over': True,\n",
       "  u'parts': True,\n",
       "  u'people': True,\n",
       "  u'picking': True,\n",
       "  u'pink': True,\n",
       "  u'power': True,\n",
       "  u'pretty': True,\n",
       "  u'quick': True,\n",
       "  u're': True,\n",
       "  u'real': True,\n",
       "  u'really': True,\n",
       "  u'regarding': True,\n",
       "  u'review': True,\n",
       "  u'robot': True,\n",
       "  u'robots': True,\n",
       "  u'russian': True,\n",
       "  u's': True,\n",
       "  u'schnazzy': True,\n",
       "  u'sequences': True,\n",
       "  u'ship': True,\n",
       "  u'shot': True,\n",
       "  u'so': True,\n",
       "  u'some': True,\n",
       "  u'someone': True,\n",
       "  u'stan': True,\n",
       "  u'star': True,\n",
       "  u'starring': True,\n",
       "  u'start': True,\n",
       "  u'still': True,\n",
       "  u'story': True,\n",
       "  u'strangeness': True,\n",
       "  u'stumbling': True,\n",
       "  u'substance': True,\n",
       "  u'sunken': True,\n",
       "  u'sutherland': True,\n",
       "  u't': True,\n",
       "  u'tech': True,\n",
       "  u'that': True,\n",
       "  u'the': True,\n",
       "  u'there': True,\n",
       "  u'these': True,\n",
       "  u'they': True,\n",
       "  u'thing': True,\n",
       "  u'this': True,\n",
       "  u'throughout': True,\n",
       "  u'time': True,\n",
       "  u'to': True,\n",
       "  u'took': True,\n",
       "  u'tugboat': True,\n",
       "  u'turn': True,\n",
       "  u'very': True,\n",
       "  u'virus': True,\n",
       "  u'was': True,\n",
       "  u'wasted': True,\n",
       "  u'we': True,\n",
       "  u'well': True,\n",
       "  u'what': True,\n",
       "  u'when': True,\n",
       "  u'why': True,\n",
       "  u'william': True,\n",
       "  u'winston': True,\n",
       "  u'with': True,\n",
       "  u'within': True,\n",
       "  u'work': True,\n",
       "  u'y2k': True,\n",
       "  u'you': True,\n",
       "  u'your': True},\n",
       " 'neg')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "negreviews[1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to evaluate our algorithm at a later step, we will need to split the dataset into training and test set. \n",
    "\n",
    "Here, we will use 75% of the data as the training set, and the rest as the test set. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "negsplit = int(len(negreviews)*0.75)\n",
    "possplit = int(len(posreviews)*0.75)\n",
    "\n",
    "trainingset = negreviews[:negsplit] + posreviews[:possplit]\n",
    "testset = negreviews[negsplit:] + posreviews[possplit:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The classifier training method expects to be given a list of tokens in the form of [(feats, label)] where feats is a feature dictionary and label is the classification label. In our case, feats will be of the form {word: True} and label will be one of ‘pos’ or ‘neg’."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train on 1500 instances, test on 500 instances\n"
     ]
    }
   ],
   "source": [
    "print 'train on %d instances, test on %d instances' % (len(trainingset), len(testset))\n",
    " \n",
    "classifier = NaiveBayesClassifier.train(trainingset)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For accuracy evaluation, we can use nltk.classify.util.accuracy with the test set as the gold standard.\n",
    " \n",
    "Accuracy is described as follows (taken from NLTK documentation): Given a list of reference values and a corresponding list of test values, return the fraction of corresponding values that are equal. \n",
    "\n",
    "In particular, return the fraction of indices\n",
    "    ``0<i<=len(test)`` such that ``test[i] == reference[i]``.\n",
    "\n",
    "    :type reference: list\n",
    "    :param reference: An ordered list of reference values.\n",
    "    :type test: list\n",
    "    :param test: A list of values to compare against the corresponding  reference values.\n",
    "    :raise ValueError: If ``reference`` and ``length`` do not have the same length."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy: 0.728\n"
     ]
    }
   ],
   "source": [
    "print 'accuracy:', nltk.classify.util.accuracy(classifier, testset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In addition, NLTK allows us to see the most useful features:\n",
    "\n",
    "According to NLTK documentation, the most_informative_features() returns a list of the 'most informative' features used by the classifier.  For the purpose of this function, the informativeness of a feature ``(fname,fval)`` is equal to the highest value of P(fname=fval|label), for any label, divided by the lowest value of P(fname=fval|label), for any label:\n",
    "\n",
    "        |  max[ P(fname=fval|label1) / P(fname=fval|label2) ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Most Informative Features\n",
      "             magnificent = True              pos : neg    =     15.0 : 1.0\n",
      "             outstanding = True              pos : neg    =     13.6 : 1.0\n",
      "               insulting = True              neg : pos    =     13.0 : 1.0\n",
      "              vulnerable = True              pos : neg    =     12.3 : 1.0\n",
      "               ludicrous = True              neg : pos    =     11.8 : 1.0\n",
      "                  avoids = True              pos : neg    =     11.7 : 1.0\n",
      "             uninvolving = True              neg : pos    =     11.7 : 1.0\n",
      "              astounding = True              pos : neg    =     10.3 : 1.0\n",
      "             fascination = True              pos : neg    =     10.3 : 1.0\n",
      "                 idiotic = True              neg : pos    =      9.8 : 1.0\n"
     ]
    }
   ],
   "source": [
    "classifier.show_most_informative_features()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
