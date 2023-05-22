import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
#seaborn.set(style='whitegrid'); seaborn.set_context('talk')

from sklearn.datasets import load_iris

import ut

iris_data = load_iris()

# 3 Classes
dataSet = ut.im_data(4)
dataSet_3_classes = dataSet[:315, :]
X = dataSet_3_classes[:, :24]
y = dataSet_3_classes[:, 24].reshape(dataSet_3_classes.shape[0], 1)

#X_train /= np.max(X_train, axis=0)
#X_test /= np.max(X_test, axis=0)


n_samples, n_features = iris_data.data.shape
def Show_Diagram(_x ,_y,title):
    plt.figure(figsize=(10,4))
    plt.scatter(X[:,_x],
    X[:, _y], c=y, cmap=cm.viridis)
    plt.xlabel(f'Característica {_x}');
    plt.ylabel(f'Característica {_y}');
    plt.title(title)
    plt.colorbar(ticks=([0, 1, 2]));
    plt.show();

Show_Diagram(0, 1,'Sepal')
Show_Diagram(2, 3,'Petal')