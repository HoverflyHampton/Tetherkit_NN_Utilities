from log_parser.DFParser import DFLog
import numpy as np
import pandas as pd
import pathlib
import argparse

earth_radius = 6371e3

def bgu_to_tf(filename):
        df = pd.read_csv(filename)
        df.columns = df.columns.str.replace(' ', '')
        launch_idx = find_launch_index(df)
        create_gps_rads_columns(df)
        tk_lat, tk_lon =  get_lat_lon(df, launch_idx)
        
        get_gps_meter_delta(df, tk_lat, tk_lon)
        get_gps_angle_delta(df, tk_lat, tk_lon)
        get_gps_xy_delta(df)

        return make_clean_dataframe(df, 5000, data_collection_id=filename.stem)

def make_dataset_from_folder(folder_name, logging):
    csv_files = [f for f in pathlib.Path(folder_name).glob('*.csv')]
    end_df = pd.DataFrame()
    for indx, file in enumerate(csv_files):
        if logging:
            print("Starting File {} of {}".format(indx+1, len(csv_files)))
        end_df = pd.concat([end_df, bgu_to_tf(file)], ignore_index=True)

    return end_df

def serialize_dataframe(df: pd.DataFrame, output_file):
    df.to_csv(output_file)

def find_launch_index(df: pd.DataFrame, cutoff = 5000):
    return df.where(df['pow_motorCurrentTotal'].gt(cutoff)).first_valid_index()

def find_land_index(df: pd.DataFrame, cutoff = 5000):
    return df.where(df['pow_motorCurrentTotal'].gt(cutoff)).last_valid_index()

def create_gps_rads_columns(df):
    df['rtk_lat_rads'] = np.deg2rad(df['gps_gps2LatR']*10e-8, dtype=np.double)
    df['rtk_lon_rads'] = np.deg2rad(df['gps_gps2LonR']*10e-8, dtype=np.double)

def get_lat_lon(df, index):
    return (df['rtk_lat_rads'].iloc[index], df['rtk_lon_rads'].iloc[index]) 

def get_gps_meter_delta(df, tk_lat, tk_lon):
    df['delta_lat_rad'] = df['rtk_lat_rads'] - tk_lat
    df['delta_lon_rad'] = df['rtk_lon_rads'] - tk_lon

    
    df['rtk_a'] = (
        (np.sin(df['delta_lat_rad']/2.0) * np.sin(df['delta_lat_rad']/2.0)) + 
        (np.cos(tk_lat) * np.cos(df['rtk_lat_rads']) *
        (np.sin(df['delta_lon_rad']/2.0) * np.sin(df['delta_lon_rad']/2.0)))
    )
    df['rtk_c'] = (
        2 * np.arctan2(np.sqrt(df['rtk_a']), np.sqrt(1-df['rtk_a']))
    )
    df['rtk_delta_meters'] = (
        earth_radius * df['rtk_c']
    )

def get_gps_angle_delta(df, tk_lat, tk_lon):
    df['rtk_theta_y'] = np.sin(df['delta_lon_rad']) * np.cos(df['rtk_lat_rads'])
    df['rtk_theta_x'] = ((np.cos(tk_lat) * np.sin(df['rtk_lat_rads'])) - 
                   (np.sin(tk_lat) * np.cos(df['rtk_lat_rads'] * np.cos(df['delta_lon_rad'])))
                   )
    df['rtk_theta'] = np.arctan2(df['rtk_theta_y'], df['rtk_theta_x'])

def get_gps_xy_delta(df):
     df['rtk_x_m'] = np.cos(df['rtk_theta']) * df['rtk_delta_meters']
     df['rtk_y_m'] = np.sin(df['rtk_theta']) * df['rtk_delta_meters']

def make_clean_dataframe(df: pd.DataFrame, motor_cutoff: int, data_collection_id: str):

    truncated_data = df.loc[(df['pow_motorCurrentTotal'] > 5000)]

    clean_data = truncated_data.filter(['alt_tetherPosition',
                            'wnd_windSpeed',
                            'wnd_windBearing',
                            'cft_craftHeading',
                            'cft_craftPitch',
                            'cft_craftRoll',
                            'ats_imuPitch',
                            'ats_imuRoll',
                            'alt_lidarAltitude',
                            'alt_sonarAltitude',
                            'rtk_x_m',
                            'rtk_y_m'], axis=1)
    # Training Features
    clean_data.rename({'alt_tetherPosition':'Tether_Length',
                        'wnd_windSpeed':'Wind_Speed',
                        'wnd_windBearing':'Wind_Direction',
                        'cft_craftHeading':'Craft_Direction',
                        'cft_craftPitch':'Craft_Pitch',
                        'cft_craftRoll':'Craft_Roll',
                        'ats_imuPitch':'ATS_Pitch',
                        'ats_imuRoll':'ATS_Roll',
                        'alt_lidarAltitude':'Lidar_Altitude',
                        'alt_sonarAltitude':'Sonar_Altitude',
                        'rtk_x_m':'True_Local_Position_X',
                        'rtk_y_m':'True_Local_Position_Y',
                    })

    # Metadata Label
    clean_data['Dataset'] = [data_collection_id] * len(clean_data)
    return clean_data


def main():
    parser = argparse.ArgumentParser(
        prog="Hoverfly GPS Denied Data Parser",
        description="A program to create training data for the gps denied neural network using bgu log file csvs",
        epilog='Use -h to view help'
    )

    parser.add_argument('-i', '--input_dir', type=str, dest='input_dir', default='.', help="The path to the top level directory of the csv files to process")
    parser.add_argument('-o', '--output', type=str, dest='output_file', default='./hoverfly_nn_test_data.csv', help='The path to the csv file to output')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file

    print("Starting Dataset Collection")
    data_set = make_dataset_from_folder(input_dir, logging=True)
    serialize_dataframe(data_set, output_file)

if __name__ == '__main__':
    main()