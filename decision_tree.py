# decision_tree.py


import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import tree,metrics
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    result = {}
    unique_values,count_values = np.unique(x,return_counts=True)
    for key in unique_values:
        index_list = []
        for i in range(x.size):
            if(key==x[i]):
                index_list.append(i)
        result[key] = index_list
    return result

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    unique_labels, count_labels = np.unique(y,return_counts=True)
    prob = np.zeros(shape=(np.size(unique_labels),1))
    result=0
    for i in range(count_labels.shape[0]):
        prob[i]=count_labels[i]/np.size(y)
        #print(prob[i])
        result -= prob[i]*math.log(prob[i],2)
    
    return result

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    unique_attr_values,count_attr_values = np.unique(x,return_counts=True)
    result = entropy(y)
    #probx = np.zeros(shape=(np.size(unique_attr_values),1))
    probx = count_attr_values.astype('float')/x.shape[0]
    '''
    for i in range(count_attr_values.shape[0]):
        probx[i] = count_attr_values[i]/np.size(x)
        new_y = y[x==unique_attr_values[i]]
        entropy_new_y = entropy(np.array(new_y))
        cond_prob += probx[i]*entropy_new_y
    '''
    for p,v in zip(probx,unique_attr_values):
        result -= p*entropy(y[x==v])
    return result


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    if(attribute_value_pairs is None):
        attribute_value_pairs = []
        for i in range(x.shape[1]):
            prtn = partition(x[:,i])
            for key in prtn:
                attribute_value_pairs.append((i,key))   # (i,key) is the attribute value pair corresponding to feature xi and value key
        new_x = np.zeros(shape=(x.shape[0],len(attribute_value_pairs)))     # new X contains boolean column corresponding to each attribute value pair
        for i in range(len(attribute_value_pairs)):
            for j in range(new_x.shape[0]):
                if(x[j,attribute_value_pairs[i][0]]==attribute_value_pairs[i][1]):
                    new_x[j,i]=1
                else:
                    new_x[j,i]=0
        x = new_x
    unique_labels, count_labels = np.unique(y,return_counts=True)
    # first condition - pure class label
    if(unique_labels.shape[0]==1):
        return unique_labels[0]
    
    # second condition - empty attribute_value_pair array
    if len(attribute_value_pairs)==0:
        if(count_labels[0]>count_labels[1]):
            return unique_labels[0]
        else:
            return unique_labels[1]
    
    # third conidtion
    if(depth==max_depth):
        if(count_labels[0]>count_labels[1]):
            return unique_labels[0]
        else:
            return unique_labels[1]
    
    max_mutual_information_attr_value_pair = 0
    selected_attribute_value_pair = ()
    index_selected_attribute_value_pair = 0
    '''
    for i in range(len(attribute_value_pairs)):
        x_col = x[:,attribute_value_pairs[i][0]]    # attribute column of ith attribute_value_pair
        x_value = attribute_value_pairs[i][1]       # value of the column to check against
        x_new = np.zeros(shape=(x_col.shape[0],1))             # vector of 1s and 0s. 1 when value matches x_value. 0 otherwise.
        for j in range(x_col.shape[0]):
            if(x_col[j]==x_value):
                x_new[j] = 1
        cur_mutual_information = mutual_information(x_new,y)
        print(cur_mutual_information)
        if(cur_mutual_information>max_mutual_information_attr_value_pair):
            max_mutual_information_attr_value_pair=cur_mutual_information
            selected_attribute_value_pair = attribute_value_pairs[i]
            index_selected_attribute_value_pair = i
    '''
    # print(x.shape[1])
    # print(depth)
    # print(attribute_value_pairs)
    for i in range(x.shape[1]):
        cur_mutual_information = mutual_information(x[:,i],y)
        #print(str(attribute_value_pairs[i]) + str(cur_mutual_information))
        if(cur_mutual_information>max_mutual_information_attr_value_pair):
            max_mutual_information_attr_value_pair=cur_mutual_information
            selected_attribute_value_pair = attribute_value_pairs[i]        # store the current max arg
            index_selected_attribute_value_pair = i

    # left where attribute value pair condition is not satified
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    attribute_value_pairs.pop(index_selected_attribute_value_pair)  # delete the selected attribute value pair from the list
    selected_col = x[:,[index_selected_attribute_value_pair]]
    x = np.delete(x,index_selected_attribute_value_pair,1)          # delete the corresponding column from x
    for i in range(x.shape[0]):
        if(selected_col[i]==1):
            x_right.append(x[i,:])
            y_right.append(y[i])
        else:
            x_left.append(x[i,:])
            y_left.append(y[i])
    # converting lists to numpy arrays
    x_left = np.array(x_left)
    x_right = np.array(x_right)
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    # print("y")
    # print(y_left.shape)
    # print(y_right.shape)
    '''
    if(y_left.shape[0]==0):
        unique_labels_right, count_labels_right = np.unique(y_right,return_counts=True)
        if(count_labels_right[0]>count_labels_right[1]):
             return unique_labels_right[0]
        else:
             return unique_labels_right[1]
    elif(y_right.shape[0]==0):
        unique_labels_left, count_labels_left = np.unique(y_left,return_counts=True)
        if(count_labels_left[0]>count_labels_left[1]):
             return unique_labels_left[0]
        else:
             return unique_labels_left[1]

    '''
    av_list2 = []
    for av in attribute_value_pairs:
        av_list2.append(av)
        i+=1
    cur_depth = depth
    left_subtree = id3(x_left,y_left,attribute_value_pairs= attribute_value_pairs,depth = depth+1,max_depth=max_depth)
    right_subtree = id3(x_right,y_right,attribute_value_pairs= av_list2,depth = cur_depth+1,max_depth=max_depth)
    

    final_tree = {}
    final_tree[(selected_attribute_value_pair[0],selected_attribute_value_pair[1],False)] = left_subtree
    final_tree[(selected_attribute_value_pair[0],selected_attribute_value_pair[1],True)] = right_subtree
    return final_tree

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if(type(tree) is not dict):
        return tree
    # key = next(iter(tree))
    # if(key[2]==True):
    #     return predict_example(x,tree.get(key))
    # else:
    #     return predict_example(x,tree.get(next(iter(tree))))
    
    for key,subtree in tree.items():
        if(x[key[0]]==key[1]):
            if(key[2]):
                return predict_example(x,subtree)
        else:
            if(key[2]==False):
                return predict_example(x,subtree)


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return (1/y_true.size)*sum(y_true!=y_pred)
    


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def confusion_matrix(y_pred,y_tst):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(y_pred)):
        tp += (y_pred[i]==y_tst[i] and y_tst[i]==1)
        tn += (y_pred[i]==y_tst[i] and y_tst[i]==0)
        fp += (y_pred[i]!=y_tst[i] and y_tst[i]==0)
        fn += (y_pred[i]!=y_tst[i] and y_tst[i]==1)
    
    print("tp="+str(tp)+", tn="+str(tn)+", fp="+str(fp)+", fn="+str(fn))
    return


if __name__ == '__main__':
    ##### Problem a - ID3 algorithm
    file_list = [('./monks-1.train','./monks-1.test'),('./monks-2.train','./monks-2.test'),('./monks-3.train','./monks-3.test')]
    learned_trees = []
    print("-------Problem A-------")
    for filename in file_list:
        # Load the training data
        M = np.genfromtxt(filename[0], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(filename[1], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        sum_trn_err = 0
        sum_tst_err = 0
        tst_err = []
        trn_err = []
        height_list = []
        # Learn a decision tree of depth 3
        for height in range(1,11):
            height_list.append(height)
            decision_tree = id3(Xtrn, ytrn, max_depth=height)
            learned_trees.append(decision_tree)
            # visualize(decision_tree)

            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            y_pred_train = [predict_example(x,decision_tree) for x in Xtrn]
            tst_err.append(compute_error(ytst, y_pred)*100)
            trn_err.append(compute_error(ytrn,y_pred_train)*100)

        avg_trn_err = sum(trn_err)/10
        avg_tst_err = sum(tst_err)/10
        print(filename[0][2:9])
        print('Test Error = {0:4.2f}%.'.format(avg_tst_err))
        print('Train Error = {0:4.2f}%.'.format(avg_trn_err))

        plt.subplot(2,1,1)
        plt.plot(height_list,tst_err)
        plt.title(filename[0][2:9])
        plt.ylabel('Test error')
        plt.xlabel('Tree depth')
        plt.subplot(2,1,2)
        plt.ylabel('Train error')
        plt.xlabel('Tree depth')
        plt.plot(height_list,trn_err)
        plt.show()

    print()
    ##### problem b - visualize and confusion matrix
    print("-------Problem B-------")
    print("Monks 1 and tree depth=1")
    visualize(learned_trees[0])
    print()
    M = np.genfromtxt(file_list[0][1], missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load monks-1 test file
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    y_pred = [predict_example(x, learned_trees[0]) for x in Xtst]
    print("Confusion matrix for monks-1 and tree depth=1")
    confusion_matrix(y_pred,ytst)
    print()
    print("Monks 1 and tree depth=2")
    visualize(learned_trees[1])
    print()
    y_pred = [predict_example(x, learned_trees[1]) for x in Xtst]
    print("Confusion matrix for monks-1 and tree depth=2")
    confusion_matrix(y_pred,ytst)
    print()

    ###### problem c - scikit-learna and graphviz
    print("-------Problem C-------")
    clf = tree.DecisionTreeClassifier()
    M = np.genfromtxt(file_list[0][0], missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load monks-1 train file
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    clf = clf.fit(Xtrn,ytrn)
    data_graphviz = tree.export_graphviz(clf,out_file=None)
    graph = graphviz.Source(data_graphviz)
    graph.render("Monks-1")         # graph exported to Monks-1.pdf in the current folder
    print("-------------Visualized decision tree exported to Monks-1.pdf in the current folder-------------")
    print()

    M = np.genfromtxt(file_list[0][1], missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load monks-1 test file
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    y_pred = clf.predict(Xtst)
    print("Confusion matrix for Monks-1 using scikit-learn")
    tn, fp, fn, tp = metrics.confusion_matrix(ytst,y_pred).ravel()
    print("tp="+str(tp)+", tn="+str(tn)+", fp="+str(fp)+", fn="+str(fn))
    print()

    ##### problem d - new data - using breast cancer dataset
    ## ID3
    print("-------Problem D-------")
    # tree depth 1
    M = np.genfromtxt('./breast-cancer.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer train file
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    M = np.genfromtxt('./breast-cancer.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer test file
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    decision_tree = id3(Xtrn, ytrn, max_depth=1)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print("Confusion matrix for breast-cancer using our ID3 algorithm - tree depth 1")
    confusion_matrix(y_pred,ytst)
    print()
    print("Tree for breast-cancer using our ID3 algorithm - tree depth 1")
    visualize(decision_tree)
    print()

    # tree depth 2
    M = np.genfromtxt('./breast-cancer.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer train file
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    M = np.genfromtxt('./breast-cancer.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer test file
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    decision_tree = id3(Xtrn, ytrn, max_depth=2)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print("Confusion matrix for breast-cancer using our ID3 algorithm - tree depth 2")
    confusion_matrix(y_pred,ytst)
    print()
    print("Tree for breast-cancer using our ID3 algorithm - tree depth 2")
    visualize(decision_tree)
    print()


    ## scikit-learn
    clf = tree.DecisionTreeClassifier()
    M = np.genfromtxt('./breast-cancer.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer train file
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    clf = clf.fit(Xtrn,ytrn)
    data_graphviz = tree.export_graphviz(clf,out_file=None)
    graph = graphviz.Source(data_graphviz)
    graph.render("Breast-cancer")         # graph exported to Breast-cancer.pdf in the current folder
    print("-------------Visualized decision tree exported to Breast-cancer.pdf in the current folder----------------")
    print()

    M = np.genfromtxt('./breast-cancer.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)   # load breast-cancer test file
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    y_pred = clf.predict(Xtst)
    print("Confusion matrix for breast-cancer test set using scikit-learn")
    tn, fp, fn, tp = metrics.confusion_matrix(ytst,y_pred).ravel()
    print("tp="+str(tp)+", tn="+str(tn)+", fp="+str(fp)+", fn="+str(fn))
    print()
