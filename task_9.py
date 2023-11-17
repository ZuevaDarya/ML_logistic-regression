import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data_table = pd.read_csv('candy-data_task_9.csv',
                        delimiter=',', decimal='.', index_col='competitorname')
candy_data = pd.read_csv('candy-data_task_9.csv').values

winpercent_col_idx = 12
not_include_candies = ['Almond Joy', 'Dum Dums', 'Mr Good Bar']

# Исключаем конфеты из тренировочного списка
train_data = np.array(
    list(filter(lambda x: x[0] not in not_include_candies, candy_data)))

X = train_data[:, 1:winpercent_col_idx]
Y = train_data[:, winpercent_col_idx + 1].astype('int')

model = LogisticRegression(random_state=2019, solver='lbfgs')
model.fit(X, Y)

def predictClassByCandyName(candies_data, candy_name):
    candy_data = np.array(list(filter(lambda x: x[0] == candy_name, candies_data)))
    candy_name = candy_data[:, 0:1][0][0]
    candy_data = candy_data[:, 1:winpercent_col_idx]
    
    #Вероятность пинадлежности конфеты к классу 1
    predict_value = model.predict_proba(candy_data)[0][1]
    print(f'Вероятность пинадлежности конфеты {candy_name} к классу 1: {round(float(predict_value) , 3)}')

# тестовый набор данных
test_data = pd.read_csv('candy-test.csv').values
test_X = test_data[:, 1:12]
test_Y = test_data[:, 12].astype('int')

predictClassByCandyName(test_data, 'Warheads')
predictClassByCandyName(test_data, 'Sugar Babies')

threshold = 0.5

#Вероятности принадлежности конфет из тестового набора к классу 1
candy_probably_1 = model.predict_proba(test_X)[:, 1]
predict_values = (candy_probably_1 > threshold).astype('int')

tn, fp, fn, tp = metrics.confusion_matrix(test_Y, predict_values).ravel()
trp = tp / (tp + fn)
precision = (tp) / (tp + fp)

print(f'TPR для тестового набора данных: {round(float(trp), 3)}')
print(f'Precision для тестового набора данных: {round(float(precision), 3)}')

auc = metrics.roc_auc_score(test_Y, candy_probably_1)
print(f'AUC для тестового набора данных: {round(float(auc), 3)}')
