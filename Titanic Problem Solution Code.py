
#**************Importing the packages**********************
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import accuracy_score

#*****************Loding the data*****************************
df=pd.read_csv('Train.csv')
test=pd.read_csv('test.csv')
print('\n')
print ('Head :--->')
print(df.head())
print ('\n')
print('Description :--->')
print (df.describe())
print ('\n')
print('Information :--->')
print(df.info())


print ('\n')
print('Columns with no NaN values :--->')
print ('\n')
print (df.loc[:, df.notnull().all()])


print('\n')
print('Columns with any NaN values :--->')
print('\n')
print (df.loc[:,df.isnull().any()])
print ('\n')

df_age=df['Age']

print (df_age)

df_new_age=df_age.fillna(df_age.mean())
print (df_new_age)

#Creating a new data frame of Age
df_age_data=pd.DataFrame({'Age':df_new_age})
print (df_age_data)

#deleting the column of Age from df
df=df.drop('Age', 1)
print (df)

#Joing the new Age data frame in the df
df_new=pd.concat([df_age_data,df], axis=1)
print (df_new.head())

print (df_new[df_new.Age<1])

train=df_new
print train

#*******************************Performing feature engineering*******************

#*********************Visualising the data with survival***************88
def fun(feature):
    survived=train[train.Survived==1][feature].value_counts()
    dead=train[train.Survived==0][feature].value_counts()
    df=pd.DataFrame({ 'dead':dead, 'survived':survived})
    #df.index=['survived', 'dead']
    
    
    df.plot.bar()
    
fun('Sex')
fun('Pclass')
fun('SibSp')
fun('Parch')
fun('Cabin')
fun('Embarked')

#***********************COMBINING TEST AND TRAIN IN A LIST**************************************

train_test_dataset=[train, test]
print train_test_dataset

#*********************EXTRACTING THR Mr. AND Mrs. from the Name feature********************
for dataset in train_test_dataset:
    dataset['title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
print train['title'].value_counts()    
print test['title'].value_counts()


#******************************Mapping over the title***********************
#Mr:0
#Miss:1
#Mrs:2
#others:3

title_mapping={'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3, 'Col':3, 'Major':3, 'Mlle':3, 'Countess':3, 'Ms':3, 'Lady':3, 'Jonkheer':3, 'Don':3, 'Mme':3, 'Capt':3, 'Sir':3}
for dataset in train_test_dataset:
    dataset['title']=dataset['title'].map(title_mapping)

print train['title'].value_counts()    
print test['title'].value_counts()
    
print train.head(10)

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

print train.head(10)
print test.head(10)

fun('title')

#Mapping of Sex

#Male :0
#Female:1

sex_mapping={'male':0, 'female':1}
for dataset in train_test_dataset:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)

print train.head(10)
print test.head(10)    

fun('Sex')


#
print train.head()

#*********************************Age**************************

test['Age'].fillna(test.groupby('title')['Age'].transform('median'), inplace=True)

print test.head()

#Plot of Age with Survived
facet=sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()


#0:Child
#1:young
#2:adult
#3:mild age
#4:senior

#mapping of age
for dataset in train_test_dataset:
    dataset.loc[dataset['Age']<=16, 'Age']=0,
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26), 'Age']=1,
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']=2,
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']=3,
    dataset.loc[dataset['Age']>62, 'Age']=4

print train.head(10)   

fun('Age')


#***********************Embarked**********************
#filling missing values
Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()
dataframe1=pd.DataFrame([Pclass1,Pclass2,Pclass3])
dataframe1.index=['1st class', '2nd class', '3rd class']
dataframe1.plot.bar()


#filling the missing value

for dataset in train_test_dataset:
    dataset['Embarked']=dataset['Embarked'].fillna('S')

print train.head(10)


#***************mapping of Embarked****************
#S:0
#C:1
#Q:2

embarked_mapping={'S':0, 'C':1, 'Q':2}
for dataset in train_test_dataset:
    dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)
    
print train.head(10)    

#*******************Fare************************

    
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

print train.head(10)  

 
#*********************mapping of Fare***************************

for dataset in train_test_dataset:
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3



print train.head(10)   


#***************************Cabin*******************************
print train['Cabin'].value_counts() 

#Extracting only the first character of cabin
for dataset in train_test_dataset:
    dataset['Cabin']=dataset['Cabin'].str[:1]

#Plotting the cabin with respect to passenger class
Pclass1=train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2=train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3=train[train['Pclass']==3]['Cabin'].value_counts()
dataframe1=pd.DataFrame([Pclass1,Pclass2,Pclass3])
dataframe1.index=['1st class', '2nd class', '3rd class']
dataframe1.plot.bar()
 
#cabin mapping
cabin_mapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
for dataset in train_test_dataset:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)

print train.head(10)  

#filling the missing value
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

print train.head(10) 

#******************Family Size*****************
train['FamilySize']=train['SibSp']+train['Parch']+1
test['FamilySize']=test['SibSp']+train['Parch']+1


print train.head(10) 


#plot of family size with survived persons
facet=sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.show()


#familysize mapping
family_mapping={1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_dataset:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)
    
print train.head(10) 

#*************dropping the non-required features***************
features_drop=['SibSp', 'Parch', 'Ticket']
train=train.drop(features_drop, axis=1)
test=test.drop(features_drop, axis=1)

train=train.drop(['PassengerId'], axis=1)


print train.head(10) 

#**********train_data and target data******************
train_data=train.drop(['Survived'], axis=1)
target=train['Survived']

print train_data.head(10) 

print test.head(10)

test_data=test.drop(['PassengerId'], axis=1)


test_data['title']=test_data['title'].fillna(test_data['title'].mean())
test_data['FamilySize']=test_data['FamilySize'].fillna(test_data['FamilySize'].mean())




#Importing classifier Modules
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#*********************Applying cross validation*************************

K_fold=KFold(n_splits=10, shuffle=True, random_state=0)

#***************Knn*****************
clf=KNeighborsClassifier(n_neighbors=13)
scoring='accuracy'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print score

#Knn Score
print round(np.mean(score)*100)


#*****************************Decision Tree*************************
clf=DecisionTreeClassifier()
scoring='accuracy'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print score

#decision tree score
print round(np.mean(score)*100)


#*************************Random Forest************************
clf=RandomForestClassifier(n_estimators=13)
scoring='accuracy'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print score

#random forest score
print round(np.mean(score)*100)


#*****************************Naive Bayes*******************
clf=GaussianNB()
scoring='accuracy'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print  score

#GuassianNB score
print round(np.mean(score)*100)

#**************************SVM*********************
clf=SVC()
scoring='accuracy'
score=cross_val_score(clf, train_data, target, cv=K_fold, n_jobs=1, scoring=scoring )
print  score

#SVM score
print round(np.mean(score)*100)


#*************Got accuracy highest in SVM****************

#**********************************TESTING*************************
clf=SVC()
clf.fit(train_data, target)
prediction=clf.predict(test_data)

#***************************creating the submission file**********************
submission_neural_network=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':prediction})

submission_neural_network.to_csv('submission_SVM.csv', index=False)


print accuracy_score(target, prediction)


test_data=test_data.reindex(columns=['Age', 'Pclass', 'Sex','Fare','Cabin','Embarked','title','FamilySize'])








    
    

