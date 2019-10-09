from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd 
import pydotplus
from IPython.display import Image  
import matplotlib.pyplot as plt 

weather_df = pd.DataFrame()
weather_df['Outlook'] = ['sunny','sunny','overcast','rainy','rainy','rainy',
          'overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy']
weather_df['Temperature'] = ['hot','hot','hot','mild','cool','cool','cool','mild',
          'cool','mild','mild','mild','hot','mild']
weather_df['Humidity'] = ['high','high','high','high','normal','normal','normal',
          'high','normal','normal','normal','high','normal','high']
weather_df['Windy'] = ['false','true','false','false','false','true','true',
          'false','false','false','true','true','false','true']
weather_df['Play'] = ['no','no','yes','yes','yes','no','yes','no','yes','yes',
          'yes','yes','yes','no']

onehot = pd.get_dummies(
        weather_df[ ['Outlook', 'Temperature', 'Humidity', 'Windy'] ])

clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(onehot, weather_df['Play'])
print(tree.export_graphviz(clf_train, None))
#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, 
                                feature_names=list(onehot.columns.values), 
                                class_names=['Not_Play', 'Play'],
                                rounded=True, filled=True) 

graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())
