from __future__ import print_function
import jenkspy
import numpy as np
import gensim


def zero_pad(X, seq_len):
    X = [list(x) for x in X]
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def LR_convert(X, vocabulary_size):
    Y = []
    for x in X:
        y = np.zeros(vocabulary_size)
        for i in x:
            y[i] = 1
        Y.append(y)
    return np.array(Y)

def word_pad(X, word_len, word_index_mapping, act_index_mapping):
    new_X = []
    for x in X:
        new_x = []
        for act in x:
            if act == 0: 
                new_x += [0] * word_len
            else:
                words = act_index_mapping[act].split('_')
                words = [word_index_mapping[w] for w in words]
                words = words[:word_len - 1] + [0] * max(word_len - len(words), 1)
                if len(words) != word_len: sys.exit()
                new_x += words
        new_X.append(new_x)
    return np.array(new_X)


def load_embedding_word2vec(vocabulary):
    # load embedding_vectors from the word2vec
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    embedding_vectors = np.random.uniform(-1.0, 1.0, (len(vocabulary)+1, 300))
    for i in range(len(vocabulary)):
        try:
            embedding_vectors[i+1] = np.asarray(model[vocabulary[i+1]])
        except:
            pass
    return embedding_vectors

def load_LR_weights(vocabulary_size, filename):
    LR_weights = np.random.uniform(-1.0, 1.0, (vocabulary_size, 1))
    f = open(filename,'r')
    contents = f.readlines()
    for line in contents:
        vec = line.replace('\n','').split(':')
        LR_weights[int(vec[0])] = float(vec[1])
    return LR_weights

def load_mapping(filename):
    f = open(filename)
    vocabulary = {}
    for line in f:
        temp = line.rstrip().split('\t')
        vocabulary[temp[0]] = int(temp[1])
    return vocabulary, len(vocabulary)+1


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word

def get_sequence_length(M,N):
    return max([len(a) for a in M]+[len(b) for b in N])

def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]

def load_embedding(X, vocab_size):
    f = open(X, 'r')
    [_,dim] = f.readline().rstrip().split(' ')
    embedding_vectors = np.random.uniform(-1.0, 1.0, (int(vocab_size),int(dim)))
    line = f.readline()
    while line != '':
        items = line.rstrip().split(' ')
        vector = np.asarray(items[1:],dtype="float32")
        if int(items[0]) < vocab_size:
            embedding_vectors[int(items[0])] = vector
        line = f.readline()
    return embedding_vectors,int(dim)


def shuffle_batch(X, y, u):

    X, y, u = np.asarray(X), np.asarray(y), np.asarray(u)
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    u = u[indices]
    return X, y, u


def read_batch_dataset(file_name, batch_size):

    X, y, u = [],[],[]
    for i in range(30):
        for line in open(file_name):
            if len(X) == batch_size:
                #max_length = max(max([len(a) for a in X]), max_length)
                #vocab_size =  max(max([max(a) for a in X]) + 1, vocab_size)
                X_copy, y_copy, u_copy = shuffle_batch(X, y, u)
                X, y, u = [],[],[]
                yield X_copy, y_copy, u_copy
            vec = line.rstrip().split(' | ')
            label = vec[-2]
            acts = vec[-1]
            if len(vec) == 3: u.append(vec[0])
            else: u.append('0')
            y.append(1 if label == '1' else 0)
            X.append(np.asarray([int(t) for t in acts.split(' ')]))
        if X != [] and y != []:
            yield shuffle_batch(X, y, u)

def get_vocabulary_size_total(file_name):

    max_size = -1
    line_count = 0
    for line in open(file_name):
        line_count += 1
        vec = line.rstrip().split(' | ')
        acts = vec[-1]
        x = np.asarray([int(t) for t in acts.split(' ')])
        max_size = max(max(x),max_size)
    return max_size + 1, line_count

def get_sequence_length_total(file_name):

    max_length = -1
    for line in open(file_name):
        vec = line.rstrip().split(' | ')
        acts = vec[-1]
        x = np.asarray([int(t) for t in acts.split(' ')])
        max_length = max(len(x), max_length)
    return max_length


def activity_index_mapping(file_name):

    index_act_mapping = {}
    for line in open(file_name):
        [act, index] = line.rstrip().split('\t')
        index_act_mapping[int(index)] = act

    return index_act_mapping

def write_sig_act(sig_act, file_name):

    f = open(file_name, 'w')
    for line in sig_act:
        f.write(', '.join(line)+'\n')
    f.close()

def write_act_score_mapping(act_score_map, index_act_mapping, file_name):

    f = open(file_name, 'w')
    act_score_map = sorted(act_score_map.items(), key=lambda x: max(x[1]), reverse=True)
    for key,val in act_score_map:
        #if key == 0: 
        #    f.write('\t'.join(['0', str(val)])+'\n')
        #else:
        val = sorted(val, reverse=True)
        f.write('\t'.join([index_act_mapping[key], "count: "+str(len(val)), ' '.join([str(x) for x in val])])+'\n')
    f.close()

def write_act_score_mapping_jk(act_score_map, index_act_mapping, file_name, values):

    f = open(file_name, 'w')
    act_score_map = sorted(act_score_map.items(), key=lambda x: max(x[1]), reverse=True)
    index = 4
    for key,val in act_score_map:
        if val in values:
            f.write('FUNNEL '+str(index)+':'+'\n')
            index -= 1
        f.write('\t'.join([index_act_mapping[key], str(val)])+'\n')
    f.close()

def write_act_score_mapping_beta(beta, index_act_mapping, act_score_mapping_a, file_name):

    f = open(file_name, 'w')
    act_score_map = {}
    for i,val in enumerate(beta): act_score_map[i] = val
    act_score_map = sorted(act_score_map.items(), key=lambda x: x[1], reverse=True)
    for key,val in act_score_map:
        if key == 0: continue
        elif key not in act_score_mapping_a: continue
        else: f.write('\t'.join([index_act_mapping[key], str(val)])+'\n')
    f.close()     


def discover_significant_activity(index_act_mapping, alpha, act, lab, seq_len, topn):

    sig_act = []
    activities = []
    act_score = []
    for i,line in enumerate(alpha):

        line = line[:seq_len[i]-1]
        acts = act[i][:seq_len[i]-1]
        act_score.append(' '.join([str(index_act_mapping[acts[k]])+'##'+str(line[k]) for k in range(len(line))]))

        if int(lab[i]) != 1: 
            sig_act.append(["negative sample"])
            continue

        acts_copy = []
        for j,item in enumerate(acts):
            # if item == 11988:
            #     print(line[j])
            acts_copy.append(index_act_mapping[item])
        indexes = sorted(range(len(line)), key=lambda i: line[i])[-topn:][-1::-1]
        sig_act.append([index_act_mapping[act[i][j]] for j in indexes])

        for j,index in enumerate(indexes):
            if j == 0: acts_copy[index] = '->>'+acts_copy[index]+'<<-'
            else: acts_copy[index] = '->'+acts_copy[index]+'<-'
        activities.append(acts_copy)

    return sig_act, activities, act_score

def activity_score_mapping(alpha, act, lab, act_score_map_p, act_score_map_a, seq_len):

    for i,line in enumerate(alpha):
        line = line[:seq_len[i]-1]
        for j,score in enumerate(line):
            score = float("{0:.4f}".format(score))
            if act[i][j] not in act_score_map_a:
                act_score_map_a[act[i][j]] = [score]
            else: 
                act_score_map_a[act[i][j]].append(score)#max(score,act_score_map_a[act[i][j]])
        if int(lab[i]) == 0: 
            continue       
        for j,score in enumerate(line):
            score = float("{0:.4f}".format(score))
            if act[i][j] not in act_score_map_p:
                act_score_map_p[act[i][j]] = [score]
            else:
                act_score_map_p[act[i][j]].append(score)#max(score,act_score_map_p[act[i][j]])
    return act_score_map_p, act_score_map_a

def jenks_break(act_score_map, num):

    values = []
    act_score_map = sorted(act_score_map.items(), key=lambda x: x[1], reverse=True)
    for _,val in act_score_map: values.append(val)
    return jenkspy.jenks_breaks(values, nb_class=num)[1:]

def write_direct_score(u_global, embeddings, act_index_mapping, file_name):

    f = open(file_name, 'w')
    index_score_mapping = {}
    for i,embedding in enumerate(embeddings):
        if i in act_index_mapping:
            index_score_mapping[i] = np.inner(u_global, embedding)
    values = jenks_break(index_score_mapping, 4)
    index_score_mapping = sorted(index_score_mapping.items(), key=lambda x: x[1], reverse=True)
    index = 4
    for key,val in index_score_mapping:
        if val in values:
            f.write('FUNNEL '+str(index)+':'+'\n')
            index -= 1
        f.write('\t'.join([act_index_mapping[key], str(val)])+'\n')

def write_emb_pred(u_batch, pred, thr, lab, u_emb, act_scores, file_name):
    f = open(file_name, 'w')
    for i in range(len(pred)):
        emb = ' '.join([str(x) for x in u_emb[i]])
        f.write('\t'.join([str(u_batch[i]), str(pred[i]), str(thr), str(lab[i]), emb, act_scores[i]])+'\n')
    f.close()
