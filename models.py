import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing


def run(data, features):
    x = preprocessing.scale(data[features])
    y = preprocessing.scale(data['diversity'])
    feature = PolynomialFeatures()
    train_x = feature.fit_transform(x)
    test_x = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
    test_x_lr = feature.fit_transform(test_x)

    lrModel = LinearRegression()
    lrModel.fit(train_x, y)

    svrModel = SVR()
    svrModel.fit(x, y)

    knnModel = KNeighborsRegressor()
    knnModel.fit(x, y)

    if len(features) == 1:
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='b')
        ax.set_xlabel('Density')
        ax.set_ylabel('Diversity')

        ax.plot(test_x, lrModel.predict(test_x_lr), color='r')
        ax.plot(test_x, svrModel.predict(test_x), color='g')
        ax.plot(test_x, knnModel.predict(test_x), color='y')

        ax.legend(['LR(Poly)', 'SVR', 'KNN', 'Data'])
        plt.title(' '.join(features))
    print('-------' + ' '.join(features) + '-------')
    print('LR(Poly): %f' % lrModel.score(train_x, y))
    print('SVR: %f' % svrModel.score(x, y))
    print('KNN: %f' % knnModel.score(x, y))


def main():
    data = pd.read_csv('./london.csv')
    run(data, ['bus'])
    run(data, ['rail'])
    run(data, ['road'])
    run(data, ['bus', 'rail', 'road'])
    plt.show()


if __name__ == '__main__':
    main()
