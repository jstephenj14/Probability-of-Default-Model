import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import missingno as msno
from sklearn import tree
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


os.chdir('C:\\Users\\jstep\\Downloads\\Credit Risk Analytics Pack\\Probablity of Default\\Supplements')
PD_data = pd.read_excel("PD.xls")

msno.matrix(PD_data)

PD_data.housing1.value_counts()

PD_data = PD_data.reset_index()

######FILTER MISSING OUT

PD_data_nm = PD_data[PD_data["housing1"] != "."]

PD_data_nm["housing1"] = pd.to_numeric(PD_data_nm["housing1"])

features = np.asarray(PD_data.columns[(PD_data.columns != "goodbad")&
                                      (PD_data.columns != "housing1")&
                                      (PD_data.columns != "is_missing")&
                                      (PD_data.columns != "index")])

X = PD_data_nm[features]
y = PD_data_nm["housing1"]

msk = np.random.rand(len(X)) < 0.8
train_X = X[msk]
test_X = X[~msk]
train_y = y[msk]
test_y = y[~msk]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt = dt.fit(train_X,train_y)

PD_train_nm = PD_data_nm[msk]
PD_test_nm = PD_data_nm[~msk]

PD_train_nm["predicted"] = dt.predict(train_X)
pd.crosstab(PD_train_nm["predicted"],PD_train_nm["housing1"] )

PD_test_nm["predicted"] = dt.predict(test_X) 
pd.crosstab(PD_test_nm["predicted"],PD_test_nm["housing1"] )
# Excellent accuracy!

# Fit model to missing data
PD_data_m = PD_data[PD_data["housing1"] == "."]

PD_data_m["housing1"] = dt.predict(PD_data_m[features])

# Append to non-missing dataset

PD_final = PD_data_nm.append(PD_data_m)

# Now PD final is ready. 
# Build WOE buckets

duration_WOE = WOE_Binning("goodbad",PD_final[["goodbad","duration"]],sign = False)
age_WOE = WOE_Binning("goodbad",PD_final[["goodbad","age"]],sign = True)
history1_WOE = WOE_Binning("goodbad",PD_final[["goodbad","history1"]], sign = True)












######################CHECKING MISSING ITSELF
def func(row):
    if row["housing1"] == ".":
        return 1
    else:
        return 0
    
PD_data["is_missing"] = PD_data.apply(func,axis = 1)

y = PD_data["is_missing"]
features = np.asarray(PD_data.columns[(PD_data.columns != "goodbad")&
                                      (PD_data.columns != "housing1")&
                                      (PD_data.columns != "is_missing")&
                                      (PD_data.columns != "index")])
X = PD_data[features]

msk = np.random.rand(len(X)) < 0.8
train_X = X[msk]
test_X = X[~msk]
train_y = y[msk]
test_y = y[~msk]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt = dt.fit(train_X,train_y)

y_predict = dt.predict(train_X)

PD_train = PD_data[msk]

PD_train["predicted"] = dt.predict(train_X)

with open("dt_classifier.txt", "w") as f:
    f = tree.export_graphviz(dt, out_file=f)
## Then take this file to http://webgraphviz.com/


















PD_melted = pd.melt(PD_data,id_vars=["index","goodbad"])
        
g = sns.FacetGrid(PD_melted,col="goodbad",row="variable", sharex=False)
g.map(plt.hist,"value")

age_WOE = WOE_Binning("goodbad",PD_data[["goodbad","age"]],defaults_threshold = 30,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)

amount_WOE = WOE_Binning("goodbad",PD_data[["goodbad","amount"]],defaults_threshold = 20,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)
age_WOE = WOE_Binning("goodbad",PD_data[["goodbad","age"]],defaults_threshold = 30,sign = True)
duration_WOE = WOE_Binning("goodbad",PD_data[["goodbad","duration"]],defaults_threshold = 20,sign = False)
