import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Model.BoostingAttackTree import BoostingAttackTreeClassifier

#produce randomly sample
#train sample

x_train=np.random.randint(1, 9, size=(10000,10))
y_train=np.random.randint(1, 9, size=10000)
y_train=list(map(lambda x:x%2, y_train))

#test sample

x_test=np.random.randint(1, 9, size=(1000,10))

#input different base estimator for train

# attack_model=BoostingAttackTreeClassifier(base_estimator=DecisionTreeClassifier())
attack_model=BoostingAttackTreeClassifier(base_estimator=LogisticRegression())
# attack_model=BoostingAttackTreeClassifier(base_estimator=SVC())
attack_model.fit(x_train, y_train)
predict=attack_model.predict(x_test)

print(predict)


