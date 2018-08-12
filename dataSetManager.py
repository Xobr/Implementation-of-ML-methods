import random as rn
import pandas as pn
from sklearn.model_selection import train_test_split

def getLinearDataSet(count = 1000,width = 1 ,b = 0.0) -> [list,list]:
    y_train = list()
    y_test = list()
    for i in range(0,count):
        y_train.append(rn.randrange(-width,width) + i + b )
    for i in range(count, count + 100):
        y_test.append(rn.randrange(-width, width) + i + b)
    return [list(range(0,count))], y_train, [list(range(count,count + 100))], y_test

def getIrisDataSet():
    df = pn.read_csv("/home/dmytro/Programs/ML/Kaggle/ML_Algorithm_Implementation/DataSets/iris.csv", sep=',')
    trans = transform_species(df)
    train, test = train_test_split(trans, test_size = 0.2)
    x_train, y_train = iris_divide_df(train)
    x_test, y_test = iris_divide_df(test)
    return x_train, y_train, x_test, y_test

def getZeroOneDataSet(count = 1000, size = 1000):
    line = int(size/2)
    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()
    for i in range(0,count):
        x = rn.randrange(0,size)
        x_train.append(x)
        if x < line:
            y_train.append(0)
        else:
            y_train.append(1)

    for i in range(0, 100):
        x = rn.randrange(0, size)
        x_test.append(x)
        if x < line:
            y_test.append(0)
        else:
            y_test.append(1)

    return x_train, y_train, x_test, y_test

def iris_divide_df(df):
    x = list()
    for i in range(0, len(df.columns.values) - 1):
        x.append(list(df[df.columns.values[i]]))
    return x, list(df['species'])

def transform_species(df: pn.DataFrame):
    maper = dict(zip(set(df['species']), list(range(0, 4))))
    y = [maper[elem] for elem in df['species']]
    df = df.drop(['species'], axis=1)
    df['species'] = y
    return df

def get_LR_datset():
    train = pn.read_csv("/home/dmytro/Programs/ML/Kaggle/ML_Algorithm_Implementation/DataSets/trainLR.csv", sep=',')
    test = pn.read_csv("/home/dmytro/Programs/ML/Kaggle/ML_Algorithm_Implementation/DataSets/testLR.csv", sep=',')
    return [list(train["x"])], list(train["y"]), [list(test["x"])], list(test["y"])