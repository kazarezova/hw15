import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


# How to make numpy arrays from pandas dataframe
df = pd.read_csv("BRCA_pam50.tsv", sep="\t", index_col=0)
X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()

# Split into training set (80%) and test set(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=17
)

# Define and fit the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=1, weights="distance", p=2))
])
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print('общая точность (без кросс-валидации):',accuracy_score(y_pred, y_test))

sep_acc={}
for i in range(len(y_test)):
    if y_test[i] not in sep_acc:
        sep_acc[y_test[i]]=[0,0]
    sep_acc[y_test[i]][1]+=1
    if y_test[i]==y_pred[i]:
        sep_acc[y_test[i]][0]+=1
print('точность предсказаний по классам:')
for i in sep_acc:
    print(i,sep_acc[i][0]/sep_acc[i][1])

for i in sep_acc:
    sep_acc[i].append(0)
for i in y_train:
    sep_acc[i][2]+=1

# в списках [0] - число правильно определенных, [1] - число в тестовой выборке, [2] - число в тренировочной выборке
    
print(sep_acc) # дело в неравном размере классов (но не только, нп triple-neg средний по размеру, но определяется хорошо -> видимо сильно отличается)

'''
общая точность (без кросс-валидации): 0.8469945355191257
точность предсказаний по классам:
Luminal A 0.9146341463414634
Healthy 0.95
Luminal B 0.6756756756756757
Triple-negative 1.0
HER2-enriched 0.6923076923076923
Normal-like 0.0
{'Luminal A': [75, 82, 329], 'Healthy': [19, 20, 79], 'Luminal B': [25, 37, 148], 'Triple-negative': [27, 27, 107], 'HER2-enriched': [9, 13, 51], 'Normal-like': [0, 4, 18]}
'''
