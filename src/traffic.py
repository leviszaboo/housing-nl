import pandas as pd
import os

# Base path and file template
dir_path = os.path.dirname(os.path.realpath(__file__))

base_path = os.path.join(dir_path, '../data/unprocessed/traffic')
file_template = 'services-2023-{month}.csv.gz'

def load_traffic_data(month):
    month_formatted = f'{month:02}' # Format month as zero-padded 2-digit string
    path = os.path.join(base_path, file_template.format(month=month_formatted))

    data = pd.read_csv(path, compression='gzip', header=0, sep=',', quotechar='"')
    
    return data

def aggregate_traffic_data(months):
    total_station_counts = pd.Series(dtype=int)

    for month in months:
        data = load_traffic_data(month)

        station_counts = data.loc[data['Service:Completely cancelled'] == False, 'Stop:Station code'].value_counts()
        
        if total_station_counts.empty:
            total_station_counts = station_counts
        else:
            total_station_counts = total_station_counts.add(station_counts, fill_value=0)

    station_counts_df = total_station_counts.reset_index()
    station_counts_df.columns = ['station_code', 'traffic_count']

    # Manually add missing traffic data for 'NA' (Nieuw Amsterdam)
    additional_data = pd.DataFrame({'station_code': ['NA'], 'traffic_count': [27180]})
    station_counts_df = pd.concat([station_counts_df, additional_data], ignore_index=True)
    
    
    return station_counts_df

output_path = os.path.join(dir_path, '../data/output/traffic_aggregated.csv')

if __name__ == '__main__':
    months = range(1, 13)
    data_aggregated = aggregate_traffic_data(months)

    data_aggregated.to_csv(output_path, index=False)


