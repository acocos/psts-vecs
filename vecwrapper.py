import os, sys
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from gensim.models.word2vec import Word2Vec as w2v
import networkx as nx

def flatten(l):
    return [item for sublist in l for item in sublist]

def norm1d(v):
    return v / np.sqrt((v**2).sum()+1e-8)

def norm2d(v):
    return v / np.sqrt((v**2).sum(axis=1)+1e-8).reshape(-1,1)

class PPVecs():
    def __init__(self):
        self.poslist = []
        self.basedir = None
        self.vecs = None
        self.index = None
        self.revidx = {}
        self.vecsnorm = None
        self.dim = 0
        self.pp2idx = {}
        self.G = None
        self.word2pos = {}
    
    def load(self, vecdir, poslist=[]):
        self.poslist = poslist
        self.basedir = vecdir
        self.vecs, self.index = self.load_vecs()
        self.revidx = {w: i for i,w in enumerate(self.index)}
        self.vecsnorm = norm2d(self.vecs)
        self.dim = self.vecs.shape[1]
        
        self.pp2idx = {}
        for i, ppstr in enumerate(self.index):
            tgt, pp2 = self.split_ppstr(ppstr)
            if tgt not in self.pp2idx:
                self.pp2idx[tgt] = {}
            self.pp2idx[tgt][pp2] = i
        
        self.G = self.form_graph()    
        
        for i, ppstr in enumerate(self.index):
            (pos,pp1), pp2 = self.split_ppstr(ppstr)
            if pp1 not in self.word2pos:
                self.word2pos[pp1] = set()
            if pp2 not in self.word2pos:
                self.word2pos[pp2] = set()
            self.word2pos[pp1].add(pos)
            self.word2pos[pp2].add(pos)
    
    def name(self):
        return self.basedir
    
    def form_graph(self):
        G = nx.DiGraph()
        self.sep = '::'
        for key in self.index:
            pos, pp1, pp2 = key.split(' ||| ')
            G.add_edge(self.sep.join((pos,pp1)), self.sep.join((pos,pp2)))
            G.add_edge(self.sep.join((pos,pp2)), self.sep.join((pos,pp1)))
        return G
    
    def vec(self, pos, pp1, pp2, norm=False):
        v_alias = ' ||| '.join((pos, pp1, pp2))
        try:
            idx = self.revidx[v_alias]
        except KeyError:
            sys.stderr.write('Vector %s not found\n' % v_alias)
            return None
        if norm:
            return self.vecsnorm[idx]
        else:
            return self.vecs[idx]
    
    def split_ppstr(self, s):
        pos, pp1, pp2 = s.split(' ||| ')
        return (pos,pp1), pp2
    
    def load_vecs(self):
        if len(self.poslist)==0:
            vecfiles = [os.path.join(self.basedir,f) for f in os.listdir(self.basedir) 
                        if '.npy' in f]
            posorder = [os.path.basename(f).replace('.npy','') for f in vecfiles]
            idxfiles = [os.path.join(self.basedir,'%s.vocabindex.txt' % pos)
                        for pos in posorder]
            for f in idxfiles:
                if not os.path.isfile(f):
                    sys.stderr.write('Index file %s does not exist\n' % f)
                    exit(1)
        else:
            idxfiles = [os.path.join(self.basedir, '%s.vocabindex.txt' % p) 
                        for p in self.poslist]
            vecfiles = [os.path.join(self.basedir, '%s.npy' % p) 
                        for p in self.poslist]
        vecs = np.concatenate([np.load(f) for f in vecfiles], axis=0)
        index = flatten([[l.strip() for l in open(f, 'r').readlines()] 
                          for f in idxfiles])
        index = np.array(index)
        return vecs, index
    
    def paraphrases_vecs(self, pos, tgt, norm=False, side='pp1'):
        ppdct = self.pp2idx.get((pos, tgt), {})
        if len(ppdct) == 0:
            sys.stderr.write('No paraphrases for (%s, %s)\n' % (pos, tgt))
            return (None, None)
        if side=='pp1':
            pps = [' ||| '.join((pos, tgt, pp2)) for pp2 in ppdct.keys()]
        elif side=='pp2':
            pps = [' ||| '.join((pos, pp2, tgt)) for pp2 in ppdct.keys()]
        else:
            sys.stderr.write('Invalid value passed to `side`\n')
            return (None, None)
        indices = np.array([self.revidx[p] for p in pps])
        if norm:
            vecs = self.vecsnorm[indices]
        else:
            vecs = self.vecs[indices]
        return pps, vecs
    
    def nearest_neighbors(self, v, n=10):
        vsum = (v**2).sum()
        if np.abs(vsum - 1.) > 1e-8:
            v = norm1d(v)
        sims = np.dot(self.vecsnorm, v)
        neighbors = np.argsort(-sims)[:n]
        return self.index[neighbors]
    
    def cluster(self, pos, tgt, n_clus=10, side='pp2'):
        pps, ppvecs = self.paraphrases_vecs(pos, tgt, norm=True, side=side)
#         sc = SpectralClustering(n_clusters=n_clus)
        sc = KMeans(n_clusters=n_clus)
        ci = sc.fit_predict(ppvecs)
        clus = {}
        for i, cn in enumerate(ci):
            if cn not in clus:
                clus[cn] = set()
            clus[cn].add(pps[i])
        return clus
    
    def shortest_path_vecs(self, pos, src, tgt, verbose=False, norm=False,
                           return_alias=False):
        '''
        Find shortest path in PPDB graph from src.pos to tgt.pos, and return
        the associated embeddings
        If path does not exist, or src and tgt are the same, or either src/tgt
        is not in vector vocabulary, return None
        '''
        def remove_sep(s,pos):
            return s.replace('%s%s' % (pos, self.sep),'')
        if src==tgt:
            sys.stderr.write('Cannot generate path between same src/tgt (%s)\n' % (src))
            return None
        try:
            src_tagged = self.sep.join((pos,src))
            tgt_tagged = self.sep.join((pos,tgt))
            sp = nx.shortest_path(self.G, src_tagged, tgt_tagged)
            if verbose:
                sys.stderr.write('Shortest path from %s to %s (%s):\n' % (src, tgt, pos))
                sys.stderr.write(' '.join([remove_sep(s, pos) for s in sp])+'\n')
            src_alias = ' ||| '.join((pos, remove_sep(sp[0], pos), remove_sep(sp[1], pos)))
            tgt_alias = ' ||| '.join((pos, remove_sep(sp[-1], pos), remove_sep(sp[-2], pos)))
            try:
                idx_src = self.revidx[src_alias]
            except KeyError:
                sys.stderr.write('No vector for %s\n' % src_alias)
                return None
            try:
                idx_tgt = self.revidx[tgt_alias]
            except KeyError:
                sys.stderr.write('No vector for %s\n' % tgt_alias)
                return None
            if return_alias:
                return (src_alias, tgt_alias)
            if norm:
                return self.vecsnorm[idx_src], self.vecsnorm[idx_tgt]
            else:
                return self.vecs[idx_src], self.vecs[idx_tgt]
        except nx.exception.NodeNotFound:
            sys.stderr.write('One of %s or %s not in pp vector vocab\n' % (src,tgt))
            return None
        except (nx.NetworkXNoPath, nx.exception.NetworkXNoPath) as e:
            sys.stderr.write('No path from %s to %s\n' % (src,tgt))
            return None
    
    def represent_pair(self, pos, w1, w2, method='shortestpath', norm=False,
                       spbackoff=True, return_alias=False):
        '''
        Generate vector representations for the pair of words, using the paraphrase
        vectors via the provided method
        :param pos: str
        :param w1: str
        :param w2: str
        :param method: str in ['shortestpath','max']
        '''
        if method=='shortestpath':
            res = self.shortest_path_vecs(pos, w1, w2, norm=norm,
                                          return_alias=return_alias)
            if res is None:
                if spbackoff:
                    return self.represent_pair(pos, w1, w2, method='max', norm=norm,
                                               return_alias=return_alias)
            return res
        elif method=='max':
            pps1, vecs1 = self.paraphrases_vecs(pos, w1, norm=True)
            pps2, vecs2 = self.paraphrases_vecs(pos, w2, norm=True)
            if (vecs1 is None) or (vecs2 is None):
                return None
            sims = vecs1.dot(vecs2.T)
            i, j = np.unravel_index(np.nanargmax(sims), sims.shape)
            v1_alias = pps1[i]
            v2_alias = pps2[j]
            if return_alias:
                return v1_alias, v2_alias
            v1 = self.vec(*v1_alias.split(' ||| '), norm=norm)
            v2 = self.vec(*v2_alias.split(' ||| '), norm=norm)
            return v1, v2
        else:
            sys.stderr.write("method for `represent_pair` must be one of ['shortestpath','max']\n")
            return None
    
    
    def most_likely_pos(self, word, n=1):
        '''
        Most likely part of speech tag for word, based on
        number of paraphrases
        '''
        w2p = self.word2pos.get(word, None)
        if w2p is None:
            sys.stderr.write('No paraphrases for word %s\n' % word)
            return None
        poslist = list(w2p)
        pps = [self.paraphrases_vecs(p, word) for p in poslist]
        ns = [len(p[0]) if p[0] else 0 for p in pps]
        p2n = dict(zip(poslist,ns))
        maxps = sorted(p2n,key=p2n.get, reverse=True)[:n]
        return maxps
    
    
    def represent_pair_nopos(self, w1, w2, primarymethod='shortestpath', norm=False,
                             return_alias=False):
        '''
        Generate vector representations for the pair of words, when no POS is given.
        First, check if the two words share the same POS tag (from among 2 most 
        likely for each word).
        If so, match the first shared POS tag and return represent_pair with that pos
        with that pos.
        If not, just find all paraphrases for each word in any POS, and take the pair
        which maximizes cosine sim.
        '''
        w1pos = self.most_likely_pos(w1, n=2)
        w2pos = self.most_likely_pos(w2, n=2)
        if w1pos is None:
            sys.stderr.write('No paraphrases for word %s\n' % w1)
            return None
        if w2pos is None:
            sys.stderr.write('No paraphrases for word %s\n' % w2)
            return None
        if len(set(w1pos) & set(w2pos)) == 0: # no shared pos
            # get nearest for any pos
            pps1 = []
            vecs1 = []
            w1allpos = self.word2pos[w1]
            w2allpos = self.word2pos[w2]
            for pos1 in w1allpos:
                pps1_, vecs1_ = self.paraphrases_vecs(pos1, w1, norm=True)
                if pps1_ is None: continue
                pps1.extend(pps1_)
                vecs1.extend(vecs1_)
            pps2 = []
            vecs2 = []
            for pos2 in w2allpos:
                pps2_, vecs2_ = self.paraphrases_vecs(pos2, w2, norm=True)
                if pps2_ is None: continue
                pps2.extend(pps2_)
                vecs2.extend(vecs2_)
            if len(vecs1)==0 or len(vecs2)==0:
                return None
            vecs1 = np.array(vecs1)
            vecs2 = np.array(vecs2)
            sims = vecs1.dot(vecs2.T)
            i, j = np.unravel_index(np.nanargmax(sims), sims.shape)
            v1_alias = pps1[i]
            v2_alias = pps2[j]
            if return_alias:
                return v1_alias, v2_alias
            v1 = self.vec(*v1_alias.split(' ||| '), norm=norm)
            v2 = self.vec(*v2_alias.split(' ||| '), norm=norm)
            return v1, v2
        else:
            # get first match
            for p1 in w1pos:
                if p1 in w2pos:
                    pos = p1
                    break
            return self.represent_pair(pos, w1, w2, method=primarymethod, 
                                       spbackoff=True, 
                                       norm=norm,
                                       return_alias=return_alias)
            
    
    def similarity(self, pos, w1, w2, method='shortestpath'):
        '''
        Compute cosine similarity between vectors for w1 and w2, using the given
        method to choose paraphrase representations
        :param pos: str
        :param w1: str
        :param w2: str
        :param method: str in ['shortestpath','max','mean']
        '''
        if method in ['shortestpath', 'max']:
            pair = self.represent_pair(pos, w1, w2, method=method, norm=True)
            if pair is None:
                return None
            v1, v2 = pair
            return np.dot(v1,v2)
        elif method=='mean':
            pps1, vecs1 = self.paraphrases_vecs(pos, w1, norm=True)
            pps2, vecs2 = self.paraphrases_vecs(pos, w2, norm=True)
            if (vecs1 is None) or (vecs2 is None):
                return None
            sims = vecs1.dot(vecs2.T)
            return np.nanmean(sims)
        else:
            sys.stderr.write("method for `similarity` must be one of ['shortestpath','max','mean']\n")
            return None

class WTVecs():
    def __init__(self):
        self.vecfile = None
        self.model = None
        self.dim = 0
        self.word2pos = {}
    
    def load(self, vecfile):
        self.vecfile = vecfile
        self.model = w2v.load(vecfile)
        self.dim = self.model.wv.syn0.shape[-1]
        
        for wp in self.model.wv.vocab.keys():
            word, pos = self.safesplit(wp)
            if word not in self.word2pos:
                self.word2pos[word] = set()
            self.word2pos[word].add(pos)
    
    def safesplit(self, wp):
        return '_'.join(wp.split('_')[:-1]), wp.split('_')[-1]
    
    def name(self):
        return self.vecfile
    
    def vec(self, pos, word, norm=True):
        if pos is None: # for non-POS-specific models
            word_alias = '_'.join(word.split())
            try:
                v = self.model[word_alias]
            except KeyError:
                sys.stderr.write('No vector for %s\n' % word_alias)
                return None
        else:
            wt = '_'.join(('_'.join(word.split()), pos))
            try:
                v = self.model[wt]
            except KeyError:
                sys.stderr.write('No vector for %s\n' % wt)
                return None
        if norm:
            return norm1d(v)
        else:
            return v
    
    def most_likely_pos(self, word, n=1):
        '''
        Most like part of speech tag for word, based on
        occurrence frequency in training corpus.
        '''
        word = '_'.join(word.split())
        w2p = self.word2pos.get(word, None)
        if w2p is None:
            sys.stderr.write('No paraphrases for word %s\n' % word)
            return None
        poslist = list(w2p)
        counts = [self.model.wv.vocab['_'.join((word, p))].count 
                  for p in poslist]
        p2n = dict(zip(poslist,counts))
        maxps = sorted(p2n,key=p2n.get, reverse=True)[:n]
        return maxps
    
    
    def represent_pair_nopos(self, w1, w2, norm=False, primarymethod=None,
                             return_alias=False):
        '''
        Generate vector representations for the pair of words, when no POS is given.
        First, check if the two words share the same POS tag (from among 2 most 
        likely for each word).
        If so, match the first shared POS tag and return represent_pair with that pos
        with that pos.
        If not, just find all paraphrases for each word in any POS, and take the pair
        which maximizes cosine sim.
        '''
        w1 = '_'.join(w1.split())
        w2 = '_'.join(w2.split())
        w1pos = self.most_likely_pos(w1, n=2)
        w2pos = self.most_likely_pos(w2, n=2)
        if w1pos is None:
            sys.stderr.write('No paraphrases for word %s\n' % w1)
            return None
        if w2pos is None:
            sys.stderr.write('No paraphrases for word %s\n' % w2)
            return None
        if len(set(w1pos) & set(w2pos)) == 0: # no shared pos
            # get nearest for any pos
            pps1 = []
            vecs1 = []
            w1allpos = self.word2pos[w1]
            w2allpos = self.word2pos[w2]
            for pos1 in w1allpos:
                pps1.append((pos1,w1))
                vecs1.append(self.vec(pos1, w1, norm=True))
            pps2 = []
            vecs2 = []
            for pos2 in w2allpos:
                pps2.append((pos2,w2))
                vecs2.append(self.vec(pos2, w2, norm=True))
            vecs1 = np.array(vecs1)
            vecs2 = np.array(vecs2)
            sims = vecs1.dot(vecs2.T)
            i, j = np.unravel_index(np.nanargmax(sims), sims.shape)
            v1_alias = pps1[i]
            v2_alias = pps2[j]
            if return_alias:
                return v1_alias, v2_alias
            v1 = self.vec(*v1_alias, norm=norm)
            v2 = self.vec(*v2_alias, norm=norm)
            return v1, v2
        else:
            # get first match
            for p1 in w1pos:
                if p1 in w2pos:
                    pos = p1
                    break
            return self.represent_pair(pos, w1, w2, norm=norm, return_alias=return_alias)
    
    
    def represent_pair(self, pos, pp1, pp2, norm=False, method=None,
                       return_alias=False):
        v1 = self.vec(pos, pp1, norm=norm)
        v2 = self.vec(pos, pp2, norm=norm)
        if (v1 is None) or (v2 is None):
            return None
        w1_alias = '_'.join(('_'.join(pp1.split()), pos))
        w2_alias = '_'.join(('_'.join(pp2.split()), pos))
        if return_alias:
            return w1_alias, w2_alias
        return v1, v2
    
    def similarity(self, pos, w1, w2, method=None):
        v1 = self.vec(pos, w1)
        v2 = self.vec(pos, w2)
        if (v1 is None) or (v2 is None):
            return None
        v1n = norm1d(v1)
        v2n = norm1d(v2)
        return np.dot(v1,v2)
