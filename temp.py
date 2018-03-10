import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import missingno as msno

os.chdir('C:\\Users\\jstep\\Downloads\\Credit Risk Analytics Pack\\Probablity of Default\\Supplements')
PD_data = pd.read_excel("PD.xls")

msno.matrix(PD_data)

PD_data.housing1.value_counts()

PD_data = PD_data.reset_index()

### Missing Value Check

def func(row):
    if row["housing1"] == ".":
        return 1
    else:
        return 0
    
PD_data["is_missing"] = PD_data.apply(func,axis = 1)

y = PD_data["is_missing"]
features = np.asarray(PD_data.columns[(PD_data.columns != "goodbad")&(PD_data.columns != "housing1")])
X = PD_data[features]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

from sklearn.externals.six import StringIO  
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


PD_melted = pd.melt(PD_data,id_vars=["index","goodbad"])
        
g = sns.FacetGrid(PD_melted,col="goodbad",row="variable", sharex=False)
g.map(plt.hist,"value")

age_WOE = WOE_Binning("goodbad",PD_data[["goodbad","age"]],defaults_threshold = 30,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)

amount_WOE = WOE_Binning("goodbad",PD_data[["goodbad","amount"]],defaults_threshold = 20,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)
age_WOE = WOE_Binning("goodbad",PD_data[["goodbad","age"]],defaults_threshold = 30,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)
