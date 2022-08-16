from collections import Counter

def eculid_dis(x,y):
        distance = 0
        for i in range(len(x)):
            distance += (x[i]-y[i])**2
        return distance**0.5
    
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
        for test_vec in test_x:
            distance_ls = []
            for index, train_vec in enumerate(self.train_x):
                dist = self.distance_func(test_vec, train_vec)
                distance_ls.append([dist,index])
            nn_k_index = [i[1] for i in sorted(distance_ls)[:self.k]]
            
            top_k_labels = []
            for index in nn_k_index:
                top_k_labels.append(self.train_y[index])
            
            predict_label = Counter(top_k_labels).most_common(1)[0][0]
            predictions.append(predict_label)
        return predictions
    
    ##########################################################################################################
    
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

iris = load_iris()

test_n = 30
X_train = iris['data'][:-test_n]
y_train = iris['target'][:-test_n]

X_test = iris['data'][test_n:]
y_test = iris['target'][test_n:]


clf = Knn(eculid_dis)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))


'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        20
           1       0.86      1.00      0.93        50
           2       1.00      0.84      0.91        50

    accuracy                           0.93       120
   macro avg       0.95      0.95      0.95       120
weighted avg       0.94      0.93      0.93       120
'''

