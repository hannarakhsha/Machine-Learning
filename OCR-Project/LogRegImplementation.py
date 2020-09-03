import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning)

train=pd.read_csv('totalimplemented_randomized_full_dataset.csv', dtype={"fullname": object, "campus": object, "city": object, "fall_term": object,
                                            "measure_name": float, "school_name": object, "ethcat": object,
                                            "total_number": float, "ceeb_id": float, "gpa": float, "status": object})
train.head()

train.isnull()

#sns.heatmap(train.isnull())
#sns.countplot(x='gpa',hue='measure_name',data=train)
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

# train.info()

train.drop(['fullname', 'city', 'school_name', 'ceeb_id', 'status', 'ethcat', 'total_number'], axis=1, inplace=True)

train.info()

campus = pd.get_dummies(train['campus'],drop_first=True)
fall_term = pd.get_dummies(train['fall_term'],drop_first=True)
train.drop(['campus', 'fall_term'], axis=1, inplace=True)
train = pd.concat([train, campus, fall_term],axis=1)

# train.info()

X_train, X_test, y_train, y_test = train_test_split(train.drop('measure_name', axis=1),
                                                    train['measure_name'], test_size=0.20,
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)


print(classification_report(y_test, predictions))
