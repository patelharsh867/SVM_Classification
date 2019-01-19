import pandas as pd
from matplotlib import pyplot 
import numpy as np
from sklearn import svm,preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

df = pd.read_csv('./hpatel8.csv', header = None)
x1 = df[0]
x2 = df[1]
y = df[2]
target = y
data = pd.DataFrame(list(zip(x1, x2)))
labelColor = []
for i in target:
    if (i == 1):
        labelColor.append('red')
    else:
        labelColor.append('green')
pyplot.scatter(x1,x2,c=labelColor)
red_patch = mpatches.Patch(color='red', label='Label : 1')
green_patch = mpatches.Patch(color='green', label='Label : 2')
pyplot.legend(handles=[red_patch,green_patch])
#pyplot.show()

x1_scale = normalize(data, axis=0, norm='max')
C_range = 2. ** np.arange(-5, 16,2)
print(C_range)
gamma_range = 2. ** np.arange(-15, 4,2)
print(gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(x1_scale, target)
score_dict = grid.cv_results_
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100,c = score_dict['mean_test_score']*100)
ax.set_xlabel("Gamma")
ax.set_ylabel("C")
ax.set_zlabel("Accuracy Score (Percent)")
#pyplot.show()
gridSearchUnrefined = pd.DataFrame(list(zip(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100)))
result = gridSearchUnrefined.sort_values(2, ascending=False)
print(grid.best_params_,grid.best_score_)
C_range = sorted(np.unique(np.array(result[1][:8])))
print(C_range)
gamma_range =sorted(np.unique(np.array(result[0][:8])))
print(gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
refinedGrid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
refinedGrid.fit(x1_scale, target)
score_dict = refinedGrid.cv_results_
#print(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100)
print(refinedGrid.best_params_,refinedGrid.best_score_)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100,c = score_dict['mean_test_score']*100)
ax.set_xlabel("Gamma")
ax.set_ylabel("C")
ax.set_zlabel("Accuracy Score (Percent)")
#pyplot.show()

gridSearchRefined = pd.DataFrame(list(zip(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100)))
refinedResult = gridSearchRefined.sort_values(2, ascending=False)
print(refinedResult)
C_range = sorted(np.unique(np.array(result[1][:4])))
print(C_range)
gamma_range =sorted(np.unique(np.array(result[0][:4])))
print(gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
secondRefinedGrid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
secondRefinedGrid.fit(x1_scale, target)
score_dict = secondRefinedGrid.cv_results_
#print(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100)
print("Best Values of C and Gamma")
print(secondRefinedGrid.best_params_)
print("Accuracy Score (Percent)")
print(secondRefinedGrid.best_score_ * 100)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(score_dict['param_gamma'], score_dict['param_C'], score_dict['mean_test_score']*100,c = score_dict['mean_test_score']*100)
ax.set_xlabel("Gamma")
ax.set_ylabel("C")
ax.set_zlabel("Accuracy Score (Percent)")
pyplot.show()