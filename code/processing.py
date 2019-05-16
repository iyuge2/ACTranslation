# -*- coding: utf-8 -*-
import os
import jieba
import json
import pickle

cur_path = os.path.dirname(__file__)

# load new dict for ancient chinese
def fenci(files):
    """
    files: ['train_origin', 'train_target', 'test_origin']
    """
    jieba.load_userdict(dict_path)
    for f in files:
        src_path = os.path.join(cur_path, '../data/' + f + '.txt')
        dst_path = os.path.join(cur_path, '../data/' + f + '_fenci.txt')
        with open(dst_path, 'w', encoding='utf-8') as tf:
            with open(src_path, 'r', encoding='utf-8') as sf:
                lines = sf.readlines()
                res = [jieba.cut(line.strip()) for line in lines]
                for words in res:
                    tf.write(' '.join(words) + '\n')

def extract_word_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = list(set([word for line in data.split('\n') for word in line.split(' ')]))
    # remember add special_words
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

def getIds(mode, save=True):
    """
    mode: train_origin / train_target
    """
    src_file = os.path.join(cur_path, '../data/processing/' + mode + '_fenci.txt')
    # extract_word_vocab
    with open(src_file, 'r', encoding='utf-8') as f:
        data = f.read() 
    int_to_word, word_to_int = extract_word_vocab(data)
    # convert words into ids
    source_int = [[word_to_int.get(word, word_to_int['<UNK>']) 
               for word in line.strip().split(' ')] for line in data.split('\n')]
    # save
    if(save):
        with open(os.path.join(cur_path, '../data/processing/' + mode + '_word2int.pickle'), 'wb') as f:
            pickle.dump(word_to_int, f)

        with open(os.path.join(cur_path, '../data/processing/' + mode + '_int2word.pickle'), 'wb') as f:
            pickle.dump(int_to_word, f)

        with open(os.path.join(cur_path, '../data/processing/' + mode + '_ids.pickle'), 'wb') as f:
            pickle.dump(source_int, f)

    return word_to_int, int_to_word, source_int

if __name__ == '__main__':
    dict_path = os.path.join(cur_path, '../dict/dict.txt')
    # segment
    fenci(['train_target', 'test_origin'])
    # process train corpus
    source_word_to_int, source_int_to_word, source_int = getIds('train_origin')
    target_word_to_int, target_int_to_word, target_int = getIds('train_target')
    # process test corpus
    src_file = os.path.join(cur_path, '../data/processing/test_origin_fenci.txt')
    with open(src_file, 'r', encoding='utf-8') as f:
        data = f.read()
    # convert words into ids
    test_int = [[source_word_to_int.get(word, source_word_to_int['<UNK>']) 
               for word in line.strip().split(' ')] for line in data.split('\n')]
    with open(os.path.join(cur_path, '../data/processing/test_origin_ids.pickle'), 'wb') as f:
            pickle.dump(source_int, f) 