import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

def run(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = data[['road', 'bus', 'rail']]
    x_scaled = min_max_scaler.fit_transform(x)
    dataset = pd.DataFrame(x_scaled)
    dataset['HI_score'] = data['diversity']

    xs = dataset[0]
    ys = dataset[1]
    zs = dataset[2]
    target = dataset['HI_score']
    plot_3d_scatter(xs, ys, zs, target, xlabel='Road Density', ylabel='Bus Density', zlabel='Rail Density')

def plot_3d_scatter(x,y,z, cs, xlabel, ylabel, zlabel, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    #plt.savefig('3d_plot.png')
    plt.show()

def main():
    data = pd.read_csv('./london.csv')
    run(data)


if __name__ == '__main__':
    main()


