import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def run(map_df, df):
    variable_list = ['HI_score', 'ROAD_DENS', 'BUS_DENS', 'RAIL_DENS']
    title_list = ['Diversity score ', 'Road density ', 'Bus density ', 'Rail density ']
    for i in range(len(variable_list)):
        variable = variable_list[i]
        data_for_map = df[['NAME', variable]]
        data_for_map = data_for_map.sort_values(by=['NAME'], ascending=True)
        merged = map_df.set_index('NAME').join(data_for_map.set_index('NAME'))
        vmin, vmax = 0, 1
        fig, ax = plt.subplots(1, figsize=(10, 6))
        merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        title = title_list[i] + 'in London Boroughs'
        ax.set_title(title)
        ax.axis('off')
        #plt.savefig('hi_score.png')
        plt.show()


def main():
    map_df = gpd.read_file('./esri_files/London_Borough_Excluding_MHW.shp')
    df = pd.read_csv('london_withCOL.csv')
    run(map_df, df)


if __name__ == '__main__':
    main()