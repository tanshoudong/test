from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
x_train=np.random.randint(1, 9, size=(10000,10))
y_train=np.random.randint(1, 9, size=10000)
y_train=list(map(lambda x:x%2, y_train))
x_test=np.random.randint(1, 9, size=(1000,10))
ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
ada.fit(x_train,y_train)
print(ada.predict(x_test))
