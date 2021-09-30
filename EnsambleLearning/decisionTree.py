from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
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

clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

d=clf.predict(X_test)

print(classification_report(y_test,d))
#avg_acc+=sum([1 if d[i]==y_test[i] else 0 for i in range(len(d))])/len(d)
#print("Average Accuracy for 10 tests:",avg_acc/10)

tree_para = {'criterion':['gini','entropy'],'max_depth':[x for x in range(5,200)] }
  
grid = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
  
# fitting the model for grid search
grid.fit(X_train, y_train)

grid_predictions=grid.predict(X_test)

print(classification_report(y_test, grid_predictions))
print(grid.best_params_)
