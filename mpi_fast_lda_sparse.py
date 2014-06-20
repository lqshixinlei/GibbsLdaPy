#!/usr/local/bin/python2.7
#coding=utf-8
import random
from  scipy import sparse
import pickle
from mpi4py import MPI

class lda(object):
  def __init__(self, topics=None, alpha=None, beta=None, itr=None, comm=None, rank=None):
    self.topics = topics
    self.alpha = alpha
    self.beta = beta
    self.itr = itr
    self.comm = comm
    self.rank = rank
    self.word_id_dict = {}
    self.id_word_dict = {}
    self.trn_data = []
    self.z = []
    self.nksum = []
    self.ndsum = []
    self.p = []

  def read_words_dict(self, words_dict_file):
    word_id = 0
    f = open(words_dict_file)
    for line in f.readlines():
      word = line.strip()
      if word not in self.word_id_dict:
        self.word_id_dict[word] = word_id
        self.id_word_dict[word_id] = word
        word_id += 1
    self.word_dict_sz = len(self.word_id_dict)

  def read_trn_data(self, trn_data_file):
    word_id = 0
    m = 0
    f = open(trn_data_file)
    for line in f.readlines():
      line = line.strip()
      words_lst = line.split(" ")
      self.trn_data.append([])
      for i in xrange(len(words_lst)):
        self.trn_data[m].append(self.word_id_dict[words_lst[i]])
      m += 1
    f.close()

  def init_est(self):
    for m in xrange(len(self.trn_data)):
      self.z.append([])
      N = len(self.trn_data[m])
      for n in xrange(N):
        self.z[m].append(0)
    self.nksum = [0 for i in xrange(self.topics)]
    print "init nwk"
    self.nwk = sparse.lil_matrix((self.word_dict_sz, self.topics))
    print self.rank,self.nwk.shape
    self.phi = sparse.lil_matrix((self.topics, self.word_dict_sz))
    self.nwkp = sparse.lil_matrix((self.word_dict_sz, self.topics))
    for i in xrange(self.topics):
      self.p.append(0)
    M = len(self.z)
    self.ndsum = [0 for i in xrange(M)]
    print "init ndk"
    self.ndk = sparse.lil_matrix((M, self.topics))
    self.theta = sparse.lil_matrix((M, self.topics))
    for m in xrange(M):
      N = len(self.z[m])
      for n in xrange(N):
        w = self.trn_data[m][n]
        self.z[m][n] = int(random.random()*self.topics)
        w_topic = self.z[m][n]
        self.nwkp[w, w_topic] += 1
        self.ndk[m, w_topic] += 1
        '''self.nksum[w_topic] += 1'''
      self.ndsum[m] += N
    print "init estimate complete!"

  def compute_nwkp(self):
    self.nwkp = sparse.lil_matrix((self.word_dict_sz, self.topics))
    M = len(self.z)
    for m in xrange(M):
      N = len(self.z[m])
      for n in xrange(N):
        w_topic = self.z[m][n]
        w = self.trn_data[m][n]
        self.nwkp[w, w_topic] += 1

  def compute_nksum(self):
    self.nksum = [0 for i in xrange(self.topics)]
    K = self.topics
    for k in xrange(K):
      V = self.word_dict_sz
      for v in xrange(V):
        self.nksum[k] += self.nwk[v,k]

  def compute_s(self):
    K = self.topics
    s = 0
    for k in xrange(K):
      s += self.alpha*self.beta / (self.beta*self.word_dict_sz+self.nksum[k])
    return s

  def compute_r(self, doc):
    K = self.topics
    r = 0
    d_t = self.ndk.getrow(doc).tocoo()
    nz_col = d_t.col
    nnz = len(nz_col)
    for k in xrange(nnz):
      r += self.ndk[doc, nz_col[k]]*self.beta / (self.beta*self.word_dict_sz+self.nksum[nz_col[k]])
    return r

  def compute_q(self, doc, word):
    K = self.topics
    q = 0
    w_t = self.nwk.getrow(word).tocoo()
    nz_col = w_t.col
    nnz = len(nz_col)
    for k in xrange(nnz):
      q += (self.alpha+self.ndk[doc, k])*self.nwk[word, nz_col[k]] / (self.beta*self.word_dict_sz+self.nksum[nz_col[k]])
    return q

  def estimate(self):
    self.comm.send(pickle.dumps(self.nwkp), dest=0, tag=10000)
    self.nwk = pickle.loads(self.comm.bcast(pickle.dumps(self.nwk), root=0))
    self.compute_nksum()
    self.comm.Barrier()
    print rank, "start iter"
    for i in xrange(self.itr):
      '''print "iteration:",i'''
      M = len(self.z)
      for m in xrange(M):
        '''print self.rank, "sampling document m:",m'''
        N = len(self.z[m])
        for n in xrange(N):
          topic = self.sampling(m, n)
          self.z[m][n] = topic
      self.compute_nwkp()
      self.comm.send(pickle.dumps(self.nwkp), dest=0, tag=i)
      print i, self.rank, "send bcast"
      self.nwk = pickle.loads(self.comm.bcast(pickle.dumps(self.nwk), root=0))
      self.comm.Barrier()

  def sampling(self, m, n):
    topic = self.z[m][n]
    w = self.trn_data[m][n]
    K = self.topics
    V = self.word_dict_sz
    self.nwk[w, topic] -= 1
    self.ndk[m, topic] -= 1
    self.nksum[topic] -= 1
    self.ndsum[m] -= 1
    for k in xrange(K):
      self.p[k] = ((self.nwk[w,k]+self.beta)/(self.nksum[k]+V*self.beta)) * \
          (self.ndk[m,k]+self.alpha)
    for k in xrange(1,K):
      self.p[k] += self.p[k-1]
    u = random.random() * self.p[K-1]
    for topic in xrange(K):
      if self.p[topic] > u:
        break;
    self.nwk[w, topic] += 1
    self.ndk[m, topic] += 1
    self.nksum[topic] += 1
    self.ndsum[m] += 1
    return topic

  def fast_estimate(self):
    self.comm.send(pickle.dumps(self.nwkp), dest=0, tag=10000)
    self.nwk = pickle.loads(self.comm.bcast(pickle.dumps(self.nwk), root=0))
    self.compute_nksum()
    self.comm.Barrier()
    print rank, "start iter"
    for i in xrange(self.itr):
      '''print "iteration:",i'''
      M = len(self.z)
      self.s = self.compute_s()
      for m in xrange(M):
        '''print self.rank, "sampling document m:",m'''
        N = len(self.z[m])
        self.r = self.compute_r(m)
        for n in xrange(N):
          topic = self.fast_sampling(m, n)
          self.z[m][n] = topic
      self.compute_nwkp()
      print i, self.rank, "start send bcast"
      self.comm.send(pickle.dumps(self.nwkp), dest=0, tag=i)
      self.nwk = pickle.loads(self.comm.bcast(pickle.dumps(self.nwk), root=0))
      print i, self.rank, "send bcast complete"
      self.comm.Barrier()

  def fast_sampling(self, m, n):
    topic = self.z[m][n]
    w = self.trn_data[m][n]
    K = self.topics
    V = self.word_dict_sz
    self.nwk[w, topic] -= 1
    self.ndk[m, topic] -= 1
    self.nksum[topic] -= 1
    self.ndsum[m] -= 1
    self.q = self.compute_q(m, w)
    u = random.uniform(0, self.s+self.r+self.q)
    if u < self.s:
      curr_sum = 0
      for k in xrange(K):
        curr_sum += self.alpha*self.beta / (self.beta*V+self.nksum[k])
        if curr_sum > u:
          topic = k
          break
    elif u > self.s and u < self.s+self.r:
      curr_sum = self.s
      d_t = self.ndk.getrow(m).tocoo()
      nz_col = d_t.col
      nnz = len(nz_col)
      for k in xrange(nnz):
        curr_sum += self.ndk[m, nz_col[k]]*self.beta / (self.beta*V+self.nksum[nz_col[k]])
        if curr_sum > u:
          topic = nz_col[k]
          break
    elif u > self.s+self.r:
      curr_sum = self.s + self.r
      w_t = self.nwk.getrow(w).tocoo()
      nz_col = w_t.col
      nnz = len(nz_col)
      for k in xrange(nnz):
        curr_sum += self.nwk[w, nz_col[k]]*(self.alpha+self.ndk[m, nz_col[k]]) / (self.beta*V+self.nksum[nz_col[k]])
        if curr_sum > u:
          topic = nz_col[k]
          break

    self.nwk[w, topic] += 1
    self.ndk[m, topic] += 1
    self.nksum[topic] += 1
    self.ndsum[m] += 1
    return topic

  def compute_theta(self):
    M = len(self.z)
    K = self.topics
    for m in xrange(M):
      for k in xrange(K):
        self.theta[m,k] = (self.ndk[m,k]+self.alpha) / (self.ndsum[m] + K*self.alpha)

  def compute_phi(self):
    print "compute phi"
    if self.rank == 0:
      self.compute_nksum()
    K = self.topics
    V = self.word_dict_sz
    for k in xrange(K):
      for w in xrange(V):
        self.phi[k,w] = (self.nwk[w,k]+self.beta) / (self.nksum[k] + V*self.beta)

  def save_model_twords(self):
    print "save model"
    K = self.topics
    V = self.word_dict_sz
    for k in xrange(K):
      word_prob_lst_tmp = []
      word_prob_lst = []
      print "Topic", k
      for v in xrange(V):
        word_prob_lst_tmp.append([v,self.phi[k,v]])
      word_prob_lst = sorted(word_prob_lst_tmp, key=lambda x:x[1], reverse=True)
      for i in xrange(10):
        print "\t",self.id_word_dict[word_prob_lst[i][0]]

  def master_proc(self):
    np = self.comm.Get_size()
    self.nwk = sparse.lil_matrix((self.word_dict_sz, self.topics))
    self.phi = sparse.lil_matrix((self.topics, self.word_dict_sz))
    print self.nwk.shape
    for p in xrange(1, np):
      s = self.comm.recv(source=p, tag=10000)
      self.nwk = self.nwk + pickle.loads(s)
      print "init receive from process :",p
    self.comm.bcast(pickle.dumps(self.nwk), root=0)
    self.comm.Barrier()
    for i in xrange(self.itr):
      print "iteration:",i
      self.nwk = sparse.lil_matrix((self.word_dict_sz, self.topics))
      for p in xrange(1, np):
        self.nwk = self.nwk + pickle.loads(self.comm.recv(source=p, tag=i))
        print "receive from process :",p
      self.comm.bcast(pickle.dumps(self.nwk), root=0)
      self.comm.Barrier()
    self.compute_phi()
    self.save_model_twords()


if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  part = "%3.3d" %(rank-1)
  trn_data = "./data/train_data_"+ part
  words_dict = "words_dict"
  lda_model = lda(100, 0.05, 0.05, 300, comm, rank)
  if rank == 0:
    lda_model.read_words_dict(words_dict)
    lda_model.master_proc()
    pass
  else:
    lda_model.read_words_dict(words_dict)
    lda_model.read_trn_data(trn_data)
    lda_model.init_est()
    lda_model.fast_estimate()
    pass
