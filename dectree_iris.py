from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from IPython.display import Image
import numpy as np  
import matplotlib.pyplot as plt 

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

#tree.plot_tree(clf.fit(iris.data, iris.target)) 

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=['spl.L','spl.W','ptl.L','ptl.W'], 
                                class_names=['Setosa', 'Versicolor','Virginica'],
                                rounded=True, filled=True) 

graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())
#graph.write_png("iris_tree.png")
new_inst = np.zeros([1,4]) # create a new instance to classify
new_inst[0,3] = .7; 

