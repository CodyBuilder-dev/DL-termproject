# -*- coding: utf-8 -*-

import numpy as np
import pickle

def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


class CIFAR_10(object):
    def __init__(self):
        self.file_1 = './data_batch_1' 
        self.file_2 = './data_batch_2'

        with open(self.file_1, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            d_decode = {}
            
            for k,v in d.items():
                d_decode[k.decode('utf-8')] = v
            d = d_decode

        with open(self.file_2, 'rb') as fo:
            d_v = pickle.load(fo, encoding='bytes')
            d_v_decode = {}
            
            for k,v in d_v.items():
                d_v_decode[k.decode('utf-8')] = v
            d_v = d_v_decode

    
        self.all_data = d['data']
        self.all_data = d['data'].reshape(d['data'].shape[0], 3, 32, 32)    

        self.all_data_v = d_v['data']
        self.all_data_v = d_v['data'].reshape(d['data'].shape[0], 3, 32, 32)    

        self.all_label = d['labels']
        self.all_label_v = d_v['labels']
        
    
    def next(self, set_name, num):
       
        if set_name == 'Train' : 

            data = self.all_data
            labels = self.all_label
    
            idx = np.arange(0, len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[i] for i in idx]
            labels_shuffle = [labels[i] for i in idx]

        elif set_name =='Validation' : 
            
            data = self.all_data_v[8000:]
            labels = self.all_label_v[8000:]
    
            idx = np.arange(0 , len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[i] for i in idx]
            labels_shuffle = [labels[i] for i in idx]

        else : 
            print('invalid set type!')

        data_shuffle = np.asarray(data_shuffle)
        data_shuffle = data_shuffle.transpose( 0, 2, 3, 1)
        
        labels_shuffle = one_hot(np.asarray(labels_shuffle), 10)

        return data_shuffle, labels_shuffle 



if __name__ == '__main__':
    dataset = CIFAR_10()


    print(dataset.next('Train', 1))
