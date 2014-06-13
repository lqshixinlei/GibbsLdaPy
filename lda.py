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
    self.theta = []
    self.phi = []

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
    for m in xrange(len(self.trn_data)):
      self.z.append([])
      N = len(self.trn_data[m])
      for n in xrange(N):
        self.z[m].append(0)
    self.nksum = [0 for i in xrange(self.topics)]
    for i in xrange(self.topics):
      self.nkw.append([0 for i in xrange(len(self.words_set))])
      self.phi.append([0 for i in xrange(len(self.words_set))])
      self.p.append(0)
    M = len(self.z)
    self.ndsum = [0 for i in xrange(M)]
    for m in xrange(M):
      self.ndk.append([0 for i in xrange(self.topics)])
      self.theta.append([0 for i in xrange(self.topics)])
    M = len(self.z)
    for m in xrange(M):
      N = len(self.z[m])
      for n in xrange(N):
        w = self.word_id_dict[self.trn_data[m][n]]
        self.z[m][n] = int(random.random()*self.topics)
        w_topic = self.z[m][n]
        self.nkw[w_topic][w] += 1
        self.ndk[m][w_topic] += 1
        self.nksum[w_topic] += 1
      self.ndsum[m] += N

  def estimate(self):
    for i in xrange(self.itr):
      print "iteration:",i
      M = len(self.z)
      for m in xrange(M):
        N = len(self.z[m])
        for n in xrange(N):
          topic = self.sampling(m, n)
          self.z[m][n] = topic

  def sampling(self, m, n):
    topic = self.z[m][n]
    w = self.word_id_dict[self.trn_data[m][n]]
    K = self.topics
    V = len(self.words_set)
    self.nkw[topic][w] -= 1
    self.ndk[m][topic] -= 1
    self.nksum[topic] -= 1
    self.ndsum[m] -= 1
    for k in xrange(K):
      self.p[k] = ((self.nkw[k][w]+self.beta)/(self.nksum[k]+V*self.beta)) * \
          ((self.ndk[m][k]+self.alpha)/(self.ndsum[m]+K*self.alpha))
    for k in xrange(1,K):
      self.p[k] += self.p[k-1]
    u = random.random() * self.p[K-1]
    for topic in xrange(K):
      if self.p[topic] > u:
        break;
    self.nkw[topic][w] += 1
    self.ndk[m][topic] += 1
    self.nksum[topic] += 1
    self.ndsum[m] += 1

    return topic

  def compute_theta(self):
    M = len(self.z)
    K = self.topics
    for m in xrange(M):
      for k in xrange(K):
        self.theta[m][k] = (self.ndk[m][k]+self.alpha) / (self.ndsum[m] + K*self.alpha)

  def compute_phi(self):
    K = self.topics
    V = len(self.words_set)
    for k in xrange(K):
      for w in xrange(V):
        self.phi[k][w] = (self.nkw[k][w]+self.beta) / (self.nksum[k] + V*self.beta)

  def save_model_twords(self):
    K = self.topics
    V = len(self.words_set)
    for k in xrange(K):
      word_prob_lst_tmp = []
      word_prob_lst = []
      print "Topic", k
      for v in xrange(V):
        word_prob_lst_tmp.append([v,self.phi[k][v]])
      word_prob_lst = sorted(word_prob_lst_tmp, key=lambda x:x[1], reverse=True)
      for i in xrange(10):
        print "\t",self.id_word_dict[word_prob_lst[i][0]]


if __name__ == "__main__":
  lda_model = lda("./test", 20, 0.05, 0.05, 100)
  lda_model.read_trn_data()
  lda_model.init_est()
  lda_model.estimate()
  lda_model.compute_theta()
  lda_model.compute_phi()
  #print lda_model.theta
  lda_model.save_model_twords()
  #for key in lda_model.word_id_dict:
  #  print key, lda_model.word_id_dict[key]
