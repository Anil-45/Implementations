"""Tfidf Implementation."""

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class CustomTfidfVectorizer:
  """Custom Tfidf Vectorizer"""
  def __init__(self):
    # Store idf of all different words in corpus
    self.idf_dict = {}
    # Store idx of all different sorted words in corpus 
    self.word_idx = {}
    
  def _term_freq(self, document: str):
    """Compute term frequency of a given document."""

    # Split at spaces 
    doc = list(document.split(" "))
    doc_len = len(doc)
    
    # Stores term frequency
    tf = dict(Counter(doc))
    
    for word in tf.keys():
      tf[word] = tf[word]/(doc_len*1.0)
    
    return tf

  def _inv_doc_freq(self, corpus: list):
    """Compute inverse document frequency of all words in corpus."""
    idf = {}
    corpus_len = len(corpus)
    
    # Find the number of documents in which a word occurs
    for document in corpus:
      doc = set(document.split(" "))
      for word in doc:
        if word not in idf:
          idf[word] = 1
        else:
          idf[word] += 1
    
    # Considering smoothed idf
    # idf[word] = log(N/n) where N=total number of documents
    #                            n=number of documents in which word occurs
    # smoothed_idf[word] = 1+log((1+N)/(1+n))
    # To avoid zero-division, log(0) issues 
    for word in idf.keys():
      idf[word] = 1 + np.log((1+corpus_len)/(1+idf[word]))
    
    self.idf_dict = {word: idf[word] for word in sorted(idf)}
    self.word_idx = {word: idx for idx, word in enumerate(sorted(idf))}
  
  def fit(self, corpus: list):
    """Fit the corpus."""
    self._inv_doc_freq(corpus=corpus)
  
  def transform(self, corpus: list):
    """Transform the corpus."""
    corpus_len = len(corpus)
    # Each row corresponds to a document in corpus 
    rows = []
    # Store the idx of words whose tfidf is not zero 
    columns = []
    # value of tfidf
    tfidf_values = []
    
    for row_idx, doc in enumerate(corpus):
      # Compute term frequency 
      tf = self._term_freq(doc)
      
      for word, freq in tf.items():
        if word in self.idf_dict:
          # Compute tfidf
          tfidf = freq*self.idf_dict[word]
          if 0 != tfidf:
            col_idx = self.word_idx[word]
            rows.append(row_idx)
            columns.append(col_idx)
            tfidf_values.append(tfidf)
        else:
          print(f"{word} not present in Vocabulary")
    
    # Create sparse matrix
    sparse_mat = csr_matrix((tfidf_values, (rows, columns)),
                               shape=(corpus_len, len(self.idf_dict.keys())))
    
    # Normalize sparse matrix to create unit vectors for all docs
    norm_sparse_mat = normalize(sparse_mat, norm='l2', axis=1, copy=True)
    
    return norm_sparse_mat
        
  def fit_transform(self, corpus: list):
    """Fit and transform the corpus."""
    self.fit(corpus=corpus)
    return self.transform(corpus=corpus)
  
  def get_feature_names_out(self):
    """Get all the unique words in corpus."""
    return np.array(list(self.idf_dict.keys()))
  
  
corpus = [
  'this is the first document here',
  'this document is the second document',
  'and this is the third one',
  'is this the first document',
]      

# Custom Tfidf 
cf = CustomTfidfVectorizer()
cf_mat = cf.fit_transform(corpus=corpus)
cf_feat = cf.get_feature_names_out()

# Sklearn tfidf
sf = TfidfVectorizer()
sf_mat = sf.fit_transform(corpus)
sf_feat = sf.get_feature_names_out()

print(cf_feat)
print(sf_feat)

try:
  np.testing.assert_array_equal(cf_feat, sf_feat)
  print("Features matched.")
except AssertionError:
  print("Features did not match.")

print(cf_mat)
print('--------------------')
print(sf_mat)
