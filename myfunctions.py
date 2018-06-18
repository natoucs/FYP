import re #regexp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Extract action-to-group dictionary & group labels
def action_to_group (location, family):
    
    with open(location, 'r') as f:
        pattern = '_(\w+)' #for Object by default
        if family == 'Motion': pattern = '(\w+)_' #add patterns like this as a list
        regexp = re.compile(pattern)
        get = re.findall( regexp, f.read() )

    atog = list() #action to group dictionary 
    g_labels = list() 
    for i in get:
        #check if group already exists
        if i not in g_labels: #extract group number as index of word in submitted & add it to action group dictionary
            g_labels.append(i)
        atog.append( g_labels.index(i) ) 

    #num_g = len(g_labels) FYI

    gtoa = []
    for group_number in range(max(atog)+1):
        gtoa.append( [index for index, value in enumerate(atog) if value == group_number] )
    
    return (atog, gtoa, g_labels) 
    
def get_group_labels(filename, atog, group_number):
    with open(filename) as f:
        all_45_action_labels = [word for line in f for word in line.split()]
    action_labels_inside_this_group = [all_45_action_labels[i] for i in range(len(atog)) if atog[i] == group_number]
    return action_labels_inside_this_group

def read_data(filename):
    # Reads file containing features and returns features indexed by time
    x = []
    tmp_length = 0
    with open(filename) as f:
        for line in f:
            numbers_str = line.split()
            nums_float = [float(a) for a in numbers_str]
            x.append(nums_float)
            tmp_length =tmp_length+1
            # print(len(x))
    # x.extend([[0.0]*feat_size]*(padding_size-len(x)+1))
    f.close() #necessary ? supposed to be automatic
    max_seq_l = 120
    tmp_val = np.min([tmp_length-1,max_seq_l])
    #print(tmp_val)
    return x[1:]  # ignore de first line (num of frames)

def read_config(filename):
    # Reads config file and returns filenames and class label
    x = []
    with open(filename) as f:
        for line in f:
            line_split = line.split()
            x.append(line_split)
    f.close()
    return x

# one hot encoding
def num_to_idx(num, num_classes):
    vec = np.zeros( shape=num_classes, dtype=np.float) #hardcode here
    vec[num] = 1
    return vec

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    import itertools
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    
    plt.colorbar(im,fraction=0.046, pad=0.04)

    #ax = plt.gca()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')