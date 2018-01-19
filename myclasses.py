""" Code to do exploratory common functions """
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import sklearn
import math
from matplotlib import pyplot as plt
from sklearn import metrics

class utils:

    def print_scores(self, scores):
        print(scores)
        df = pd.Series(scores)

        plt.bar(range(len(df)), df.values, align='center')
        plt.xticks(range(len(df)), df.index.values, size='small')
        plt.show()
    
    def print_metrics(self, predictions, Y_test):
        accuracy = metrics.accuracy_score(predictions,Y_test)
        precision = metrics.precision_score(predictions,Y_test,pos_label=1)
        recall = metrics.recall_score(predictions,Y_test,pos_label=1)
        f1_score = metrics.f1_score(predictions,Y_test,pos_label=1)

        print("Accuracy : " +str(accuracy))
        print("Precision : " +str(precision))
        print("recall : " +str(recall))
        print("f1 Score : " +str(f1_score))
        print()

        confusion_matrix = metrics.confusion_matrix(Y_test,predictions,labels=[1,0]);
        print(confusion_matrix);

        self.drawConfusionMatrix(confusion_matrix);        

    def drawConfusionMatrix(self, confusion_matrix):
        labels = ['Late', 'Paid']
        fig = plt.figure()

        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion_matrix)
        fig.colorbar(cax)

        plt.title('Confusion matrix')
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

class MyDecisionTree:
    
    def fit(self, trainData, attributes):
        global target , trainedTree
        self.target = 'Late_Loan';
        tree = self.id3(df = trainData, target_attribute_name = self.target, attribute_names = attributes)
        self.trainedTree = tree
        return tree

    def train_test_split(self, X ,Y, test_size):
        source = X.copy()
        newY = Y.copy()
        source['Late_Loan'] = newY
        prepared = self.prepare(source)
        return train_test_split(prepared, test_size = test_size)

    def prepare(self, source):
        
        for column in source.columns:
            if source.dtypes[column] == 'float64':
                median = source[column].median()
                source[column] = source.apply(lambda object: self.CalculateMeanBin(object[column], median),axis=1)
        return source

    def CalculateMeanBin(self, value, median):
        if value > median:
            return  ">" + str(median)
        else:
            return str(median) + "<"

    def id3(self, df, target_attribute_name, attribute_names, default_class = 1):
    
        ## Tally target attribute:
        cnt = Counter(x for x in df[target_attribute_name])
        keys = list(cnt.keys())
        ## First check: Is this split of the dataset homogeneous?
        # if yes, return that homogenous label
        if len(cnt) == 1:
            return keys[0]
        
        ## Second check: Is this split of the dataset empty?
        # if yes, return a default value
        elif df.empty or (not attribute_names):
            return default_class 
        
        ## Otherwise: This dataset is ready to be divvied up!
        else:
            print 
            # Get Default Value for next recursive call of this function:
            index_of_max = list(cnt.values()).index(max(cnt.values())) 
            default_class = index_of_max # most common value of target attribute in dataset
            
            # Choose Best Attribute to split on:
            gainz = [self.information_gain(df, attr, target_attribute_name) for attr in attribute_names]
            gainz = list(gainz)
            index_of_max = gainz.index(max(gainz)) 
            best_attr = attribute_names[index_of_max]
            
            # Create an empty tree, to be populated in a moment
            tree = {best_attr:{}}
            remaining_attribute_names = [i for i in attribute_names if i != best_attr]
            
            # Split dataset
            # On each split, recursively call this algorithm.
            # populate the empty tree with subtrees, which
            # are the result of the recursive call
            for attr_val, data_subset in df.groupby(best_attr):
                subtree = self.id3(data_subset, target_attribute_name, remaining_attribute_names)
                tree[best_attr][attr_val] = subtree
            return tree

    def entropy(self, probs):
        '''
        Takes a list of probabilities and calculates their entropy
        '''
        return sum( [-prob*math.log(prob, 2) for prob in probs] )
        
    def entropy_of_list(self, a_list):
        '''
        Takes a list of items with discrete values (e.g., poisonous, edible)
        and returns the entropy for those items.
        '''        
        # Tally Up:
        cnt = Counter(x for x in a_list)
        
        # Convert to Proportion
        num_instances = len(a_list)*1.0
        probs = [x / num_instances for x in cnt.values()]
        
        # Calculate Entropy:
        return self.entropy(probs)

    def information_gain(self, df, split_attribute_name, target_attribute_name, trace=0):
        '''
        Takes a DataFrame of attributes, and quantifies the entropy of a target
        attribute after performing a split along the values of another attribute.
        '''
        
        # Split Data by Possible Vals of Attribute:
        df_split = df.groupby(split_attribute_name)
        
        # Calculate Entropy for Target Attribute, as well as Proportion of Obs in Each Data-Split
        nobs = len(df.index) * 1.0
        df_agg_ent = df_split.agg({target_attribute_name : [self.entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
        df_agg_ent.columns = ['Entropy', 'PropObservations']
        if trace: # helps understand what fxn is doing:
            print(df_agg_ent)
        
        # Calculate Information Gain:
        new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
        old_entropy = self.entropy_of_list(df[target_attribute_name])
        return old_entropy-new_entropy

    def classify(self, instance, tree, default):
        attribute = list(tree.keys())[0]
        if instance[attribute] in list(tree[attribute].keys()):
            result = tree[attribute][instance[attribute]]
            if isinstance(result, dict): # this is a tree, delve deeper
                return self.classify(instance, result, default)
            elif result is not None:
                return result # this is a label
            else:
                return default
        else:
            return default
        
    def predict(self, testData):
        return testData.apply(lambda obj: self.classify(instance = obj, tree = self.trainedTree, default=0) , axis=1)

class MyKFold:
    def predict(self, data, model, n_splits= 10, shuffle = True):
        scores = []
        for i in range(0 , n_splits):
            sample = data.sample(frac= 0.3, replace = shuffle)
            predictions = model.predict(sample)
            precision = metrics.precision_score(predictions,sample['Late_Loan'],pos_label=1)
            scores.append(precision)
        return scores
