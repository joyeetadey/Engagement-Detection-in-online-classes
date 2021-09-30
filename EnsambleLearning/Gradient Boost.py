from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import csv

data=open('finalV3.csv')
csvreader=csv.reader(data)
header=[]
header=next(csvreader)
rows=[]
for row in csvreader:
    rows.append(row)

for x in rows:
    if "" in x:
        rows.remove(x)

X=[x[:3] for x in rows]
y=[x[3] for x in rows]
for x in X:
    x[0]=int(x[0])
    if x[1]=="Active":
        x[1]=1
    else:
        x[1]=0
    if x[2]=="Frustated":
        x[2]=0
    elif x[2]=="Engaged":
        x[2]=1
    elif x[2]=="Bored":
        x[2]=2
    else:
        x[2]=3
avg_acc=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

clf=GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train,y_train)

d=clf.predict(X_test)

print(classification_report(y_test,d))

grid = GridSearchCV(GradientBoostingClassifier(), tree_para, cv=5)
  
# fitting the model for grid search
grid.fit(X_train, y_train)

grid_predictions=grid.predict(X_test)

print(classification_report(y_test, grid_predictions))
print(grid.best_params_)
