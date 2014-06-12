#!/usr/bin/python
#coding=utf-8
import random

class lda(object):
  def __init__(self, trn_data_name=None, topics=None, alpha=None, beta=None, itr=None, model_name=None):
    self.trn_data_name = trn_data_name
    self.topics = topics
    self.alpha = alpha
    self.beta = beta
    self.itr = itr
    self.model_name = model_name
    self.words_set = set()
    self.word_id_dict = {}
    self.id_word_dict = {}
    self.trn_data = []
    self.z = []
    self.nkw = []
    self.ndk = []
    self.nksum = []
    self.ndsum = []
    self.p = []

  def read_trn_data(self):
    word_id = 0
    m = 0
    f = open(self.trn_data_name)
    for line in f.readlines():
      line = line.strip()
      words_lst = line.split(" ")
      self.trn_data.append([])
      for i in xrange(len(words_lst)):
        self.trn_data[m].append(words_lst[i])
        if words_lst[i] not in self.words_set:
          self.words_set.add(words_lst[i])
          self.word_id_dict[words_lst[i]] = word_id
          self.id_word_dict[word_id] = words_lst[i]
          word_id += 1
      m += 1
    f.close()

  def init_est(self):
    for m in xrange(len(self.z)):
      self.z.append([])
      N = len(self.z[m])
      for n in xrange(N):
        self.z[m].append(0)
    if topics != None:
      self.topics = topics
      for i in xrange(topics):
        self.nkw.append([0 for i in xrange(len(self.words_set))])
        self.p.append(0)
    M = len(self.z)
    for m in xrange(M):
      self.ndk.append([0 for i in xrange(self.topics)])
    M = len(self.z)
    for m in xrange(M):
      N = len(self.z[m])
      for n in xrange(N):
        w = word_id_dict[self.trn_data[m][n]]
        self.z[m][n] = int(random.random()*topics)
        self.nkw[z[m][n]][w] += 1
        self.ndk[m][z[m][n]] += 1
        self.nksum[z[m][n]] += 1
      self.ndsum[m] += N

  def estimate(self):
    for i in xrange(self.itr):
      M = len(self.z)
      for m in xrange(M):
        N = len(self.z[m])
        for n in xrange(N):
          topic = sampling(m, n)
          z[m][n] = topic

  def sampling(self, m, n):
    topic = z[m][n]
    w = self.word_id_dict[self.trn_data[m][n]]
    K = self.topics
    V = len(self.words_set)
    self.nkw[topic][w] -= 1
    self.ndk[m][topic] -= 1
    self.nksum[topic] -= 1
    self.ndsum[m] -= 1
    for k in xrange(K):
      self.p[k] = ((nkw[k][w]+self.beta)/(nksum[k]+V*self.beta)) * \
          ((ndk[m][k]+self.alpha)/(ndsum[m]+K*self.alpha))
    for k in xrange(1,K):
      self.p[k] += p[k-1]
    u = random.random() * p[K-1]
    for topic in xrange(K):
      if p[topic] > u:
        break;
    self.nkw[topic][w] += 1
    self.ndk[m][topic] += 1
    self.nksum[topic] += 1
    self.ndsum[m] += 1

    return topic

if __name__ == "__main__":
  lda_model = lda("./test", 5, 0.05, 0.05, 100)
  lda_model.read_trn_data()
  #for key in lda_model.word_id_dict:
  #  print key, lda_model.word_id_dict[key]
