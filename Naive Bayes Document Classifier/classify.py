#Abhay Kumar (Kumar95)
#CS540

import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # DONE: add your code here
    #with open(os.path.join(os.getcwd(),filepath), 'r') as doc:
    with open(filepath, 'r', encoding='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if word in vocab:
                if word not in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
            else:
                bow[None] = 1 if None not in bow else bow[None]+1
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    total_num_files = len(training_data)
    num_files_label1 = 0
    for files in training_data:
        if files['label'] == label_list[0]:
            num_files_label1 +=1 
    logprob[label_list[0]] = math.log((num_files_label1 + 1))-math.log((total_num_files + 2))
    logprob[label_list[1]] = math.log((total_num_files - num_files_label1 + 1))- math.log((total_num_files + 2))
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    
    vocab_word_count = {}
    for word in vocab:
        vocab_word_count[word] = 0
    vocab_word_count[None] = 0
    
    all_word_count = 0
    
    # TODO: add your code here
    vocab_size = len(vocab)
    for files in training_data:
        if files['label'] == label:
            for words in files['bow']:
                vocab_word_count[words] +=  files['bow'][words]
                all_word_count +=  files['bow'][words]
            
    for vocab_word in vocab_word_count:
        word_prob[vocab_word] = math.log(vocab_word_count[vocab_word]+1)  - math.log(all_word_count + vocab_size + 1)
    

    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    
    vocab = model['vocabulary']
    test_bow = create_bow(vocab, filepath)
    
    prob_label_2016 = model['log prior']['2016']
    prob_label_2020 = model['log prior']['2020']
    
    with open(filepath, 'r', encoding='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if word in vocab:
                prob_label_2016 += model['log p(w|y=2016)'] [word]
                prob_label_2020 += model['log p(w|y=2020)'] [word]
            else: #OOV words
                prob_label_2016 += model['log p(w|y=2016)'] [None]
                prob_label_2020 += model['log p(w|y=2020)'] [None]

# # Using below makes the output vary at the 10 decimel place    
#     for word in test_bow:
#         prob_label_2016 += model['log p(w|y=2016)'] [word] * test_bow[word]
#         prob_label_2020 += model['log p(w|y=2020)'] [word] * test_bow[word]
    retval = {}
    # TODO: add your code here
    retval['log p(y=2020|x)'] = prob_label_2020
    retval['log p(y=2016|x)'] = prob_label_2016 
    retval['predicted y'] = '2016' if prob_label_2016 > prob_label_2020 else '2020'
   
    
    return retval
