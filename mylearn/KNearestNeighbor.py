from collections import Counter

def eculid(x,y):
    distance = 0
    for i in range(len(x)):
        distance += (x[i]-y[i])**2
    return distance**0.5

def pairwise_distance(X,Y):
    distance_matrix = [[0]*len(X) for _ in range(len(Y))]
    for xi, x_vec in enumerate(X):
        for yi, y_vec in enumerate(Y):
            distance_matrix[yi][xi] = eculid(x_vec, y_vec)
    return distance_matrix

def smallest_n_index(arr, n=1):
    if n==1:
        return min(enumerate(arr), key=lambda x:x[1])[0]
    top_ls = sorted(enumerate(arr), key=lambda x:x[1])[:n]
    return [x[0] for x in top_ls]
    
class Knn:
    def __init__(self, distance_func, k=5):
        self.train_x = None
        self.k = k
        self.distance_func = distance_func
    
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
    
    def predict(self, test_x):
        predictions = []
        pairwise_distance_mat = pairwise_distance(self.train_x , test_x)
        for distance in pairwise_distance_mat:
            index_list = smallest_n_index(distance, n=self.k)
            prediction = Counter(self.train_y[idx] for idx in index_list).most_common()[0][0]
            predictions.append(prediction)    
        return predictions
    
##########################################################################################################

from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import random

random.seed(10)


iris = load_iris()
index = list(range(150))
random.shuffle(index)
iris['data'] = iris['data'][index]
iris['target'] = iris['target'][index]


split_n = 100
X_train = iris['data'][:split_n]
y_train = iris['target'][:split_n]

X_test = iris['data'][split_n:]
y_test = iris['target'][split_n:]


clf = Knn(eculid_dis)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))


'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        17
           1       1.00      0.94      0.97        16
           2       0.94      1.00      0.97        17

    accuracy                           0.98        50
   macro avg       0.98      0.98      0.98        50
weighted avg       0.98      0.98      0.98        50
'''

