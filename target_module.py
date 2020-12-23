import os
import csv
import re
import math
import numpy as np
import pandas as pd
from os.path import isfile, join
from classes_module import Target

def process_targets(base_path, population_data, target_list, parameters):
    infer_missing_values=parameters['infer_missing_values']
    min_date = parameters['min_date'];
    max_date = parameters['max_date'];
    min_lat = parameters['min_lat'];
    max_lat = parameters['max_lat'];
    max_for_uninhabited = parameters['max_for_uninhabited'];
    globals_type = parameters['globals_type'];



    target_list = filter_targets_for_latitude(target_list, min_lat, max_lat);
    target_list = filter_targets_for_date(target_list, min_date, max_date);
    if parameters['remove_not_direct_targets']:
        target_list = filter_targets_for_not_direct(target_list);
    if parameters['remove_not_exact_age_targets']:
        target_list = filter_targets_for_not_exact_age(target_list);
    if parameters['remove_not_figurative_targets']:
        target_list = filter_targets_for_not_figurative(target_list);



    globals_dir = os.path.join(base_path, "globals");
    if not os.path.exists(globals_dir):
        os.makedirs(globals_dir);
    filenames_in_globals = [f for f in os.listdir(globals_dir) if isfile(join(globals_dir,f))]

    globals_filename = (population_data.name + "_" + globals_type + "_lat_" + str(min_lat) + "-" + str(max_lat) + "_date_" + str(min_date) + "-" + str(max_date) +  "_mfu_" + str(max_for_uninhabited) + ".csv").lower().replace(" ", "_");
    globals_dataframe_path = os.path.join(globals_dir, globals_filename);
    if globals_filename not in filenames_in_globals:
        if globals_type == "Australia":
                globals_dataframe = load_bin_globals_for_australia(population_data, min_date, max_date, max_for_uninhabited)
        elif globals_type=="No equatorials":
            globals_dataframe = load_bin_globals_for_no_equatorials(population_data, min_lat, max_lat,min_date, max_date,max_for_uninhabited)
        elif globals_type == "France and Spain":
            globals_dataframe = load_bin_globals_for_francespain(population_data, min_date, max_date, max_for_uninhabited)
        elif globals_type == "All":
            globals_dataframe = load_all_globals_brute(population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited)
        elif globals_type=="Rest of world":
            globals_dataframe = load_bin_globals_for_rest_of_world(population_data, min_lat, max_lat,min_date, max_date,max_for_uninhabited)

        print("Generating populations...")
        globals_dataframe.to_csv(globals_dataframe_path, sep=";")
    else:
        print("Reading population file...")
        globals_dataframe = pd.read_csv(globals_dataframe_path, sep=";");

    ###################################################
    # Extract dataframe and save as processed targets #
    ###################################################
    # - saves extracted dataframe as <directory>_dataframe.csv
    # - saves the_globals as <directory>_globals_df.csv
    # - saves target list as <directory>_targets.csv
    processed_targets_dir = os.path.join(base_path, "processed_targets")

    if not os.path.exists(processed_targets_dir):
        os.makedirs(processed_targets_dir)

    # extract dataframe rank
    print("Extracting sites from targets...")
    new_df = extract_dataframe(globals_dataframe, target_list, population_data.time_window,infer_missing_values)
    # new_df = extract_dataframe(population_data, target_list, max_for_uninhabited)
    
    # get closest site to target
    new_df['distance'] = (new_df['latitude']-new_df['target_lat'])*(new_df['latitude']-new_df['target_lat']) + (new_df['longitude']-new_df['target_lon'])*(new_df['longitude']-new_df['target_lon'])
    new_df['rank'] = new_df.groupby(['target_lat', 'target_lon', 'period'])['distance'].rank(method="first",ascending=True);
    
    sites_df = new_df[(new_df['type'] == 's') & (new_df['rank'] < 2)];
    sites_df = sites_df.groupby(['density', 'latitude', 'longitude', 'period', 'type']).first().reset_index();
    non_sites_df = new_df[new_df['type'] == 'c'];

    dataframe = pd.concat([sites_df, non_sites_df],sort=True);

    if parameters["save_processed_targets"]:
        print("Saving sites dataframe...")
        # save dataframe in processed_targets folder
        dataframe_path = os.path.join(processed_targets_dir, parameters['results_directory'] + "_dataframe.csv")
        # WRITE PROCESSED TARGET DATEFRAME
        dataframe.to_csv(dataframe_path, sep=";",quoting=csv.QUOTE_NONNUMERIC, index=False) 

    return target_list, dataframe, globals_dataframe

def extract_dataframe(globals_dataframe, target_list, time_window,infer_missing_values):
    dataframes = []
    for key, target in target_list.iteritems():
        date_from = target.date_from
        lat_n = target.lat_nw
        lat_s = target.lat_se
        lon_e = target.lon_nw
        lon_w = target.lon_se

        latstrip_df = globals_dataframe.loc[(globals_dataframe.latitude.between(lat_s,lat_n)) & (globals_dataframe.period.between(date_from, date_from+time_window))]


        if lon_e < lon_w:
            non_sites_df = latstrip_df.loc[~latstrip_df.longitude.between(lon_e, lon_w)]
            sites_df = latstrip_df.loc[latstrip_df.longitude.between(lon_e, lon_w)]
        else:
            non_sites_df = latstrip_df.loc[~(latstrip_df.longitude.between(lon_e, 360) | latstrip_df.longitude.between(0, lon_w))]
            sites_df = latstrip_df.loc[(latstrip_df.longitude.between(lon_e, 360) | latstrip_df.longitude.between(0, lon_w))]
        #If infer_missing_values is true, fills in 0 densities for targets not found in database
        if sites_df.empty and infer_missing_values:
            adict={}
            adict.update({'density':0})
            adict.update({'latitude':lat_n})
            adict.update({'longitude':lon_e})
            adict.update({'period':date_from})
            row_df=pd.DataFrame(adict,index=[0])
            sites_df=sites_df.append(row_df)
        non_sites_df['type'] = 'c';
        non_sites_df['pseudo_type'] = 'b';
        sites_df['type'] = 's';
        sites_df['pseudo_type'] = 'a';

        df = pd.concat([non_sites_df, sites_df])
        df['target_id'] = key;
        df['target_location'] = target.location;
        df['target_date_from'] = target.date_from
        df['target_date_to'] = target.date_to
        df['target_lat'] = target.orig_lat
        df['target_lon'] = target.orig_lon
        df['is_dir'] = target.is_direct == "Yes"
        df['is_exact'] = target.age_estimation.lower() == "exact age"

        dataframes.append(df);
        if non_sites_df.empty:
            print(key);
            print(lon_e)
            print(lon_w)

    return pd.concat(dataframes);


def load_all_globals_brute(population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited):
    remove_uninhabited=False
    lat_np = population_data.lat_array
    lon_np = population_data.lon_array
    time_np = population_data.time_array
    den_np = population_data.density_array
    time_multiplier = population_data.time_multiplier
    density_multiplier = population_data.density_multiplier;

    # INCLUSIVE
    time_mask = (time_np*time_multiplier <= max_date) & (time_np*time_multiplier >= min_date);
    
    print("Min date: " + str(min_date));
    print("Max date: " + str(max_date));
    print("Min lat: " + str(min_lat));
    print("Max lat: " + str(max_lat));


    # INCLUSIVE
    latlon_mask = (lat_np <= max_lat) & (lat_np >= min_lat);

    mask = latlon_mask[np.newaxis,:] & time_mask[:, np.newaxis];

    latlon_length = len(lat_np);
    time_length = len(time_np);
    indices = np.reshape(range(latlon_length*time_length), [-1, latlon_length])
    valid_ind = indices[mask];


    # print("Latlon: " + str(latlon_length))
    # print("Time: " + str(time_length))
    # print("density: " + str(len(den_np)));
    # print(indices[time_length-1][latlon_length-1]/(latlon_length))
    # print(latlon_length*time_length/(time_length-1))

    print("Masking lat, lon, time..")
    periods = time_np[valid_ind/(latlon_length)]*time_multiplier

    valid_latlon_ind = valid_ind%latlon_length
    latitudes = lat_np[valid_latlon_ind]
    longitudes = lon_np[valid_latlon_ind]

    print("Masking densities");
    densities = den_np[mask]*density_multiplier;

    print("Generating dataframe...")
    new_df = pd.DataFrame({'density': densities, 'period': periods, 'latitude': latitudes, 'longitude': longitudes})
    print("Filtering...")
    new_df = new_df[new_df.density > max_for_uninhabited];
    print("Population dataframe generated.")

    return new_df

def load_bin_globals_for_no_equatorials (population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited):
    df = load_all_globals_brute(population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited)
    new_df = df.loc[~df.latitude.between(-10, 20)]
    return new_df

def load_bin_globals_for_australia(population_data, min_date, max_date, max_for_uninhabited):
    df = load_all_globals_brute(population_data, -40, -11, min_date, max_date, max_for_uninhabited)
    new_df = df.loc[(df.latitude.between(-39.16,-11.17)) & (df.longitude.between(112.14,154.86))]
    return new_df

def load_bin_globals_for_francespain(population_data, min_date, max_date, max_for_uninhabited):
    df = load_all_globals_brute(population_data, 34, 60, min_date, max_date, max_for_uninhabited)
    new_df = df.loc[(df.latitude.between(35.92,51.09)) & ((df.longitude.between(0, 8.12)) | (df.longitude.between(350.12, 360)))]
    return new_df

def load_bin_globals_for_rest_of_world(population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited):
    df = load_bin_globals_for_no_equatorials(population_data, min_lat, max_lat, min_date, max_date, max_for_uninhabited)
    df_minus_australia = df.loc[~((df.latitude.between(-39.16,-11.17)) & (df.longitude.between(112.14,154.86)))]
    df_rest_of_world=df_minus_australia.loc[~((df.latitude.between(35,50.84)) & ((df.longitude.between(0, 7)) | (df.longitude.between(350, 360))))]
    return df_rest_of_world
    
    
    

def read_target_list_from_csv(filename):

    target_list={}

    target_df = pd.read_csv(filename +".csv")
    for index, row in target_df.iterrows():
        location = row["Name"].replace('\n', ' ').replace('\r', '');

        lat = float(re.sub("[^0-9.-]", "", str(row["Latitude"])))
        lon = float(re.sub("[^0-9.-]", "", str(row["Longitude"])))
        if lon < 0:
            lon = lon + 360;
        lat_nw=lat+1
        lon_nw=lon-2
        lat_se=lat-1
        lon_se=lon+2

        date_from = row["Earliest age in sample"]

        date_to = row["Latest age in sample"];
        if date_to == "Modern":
            date_to = 0;
        elif date_to == "Single sample" or (not isinstance(date_to, str) and math.isnan(date_to)):
            date_to = date_from;

        country = row["Modern Country"];

        is_direct = "Yes" if row["Direct / indirect"]=="Direct" else "No";

        age_est = str(row["Exact Age / Minimum Age / Max Age"]);
        
        calibrated = "Yes" if row["Calibrated"]=="Yes" else "No";

        kind = row["Kind"]
        figurative = "No";
        if row["Figurative"] == "Unknown" or row["Figurative"] == "-":
            figurative = "No";
        elif len(figurative) > 2:
            figurative = "Yes" if row["Figurative"][0:3] == "Yes" else "No"

        # target_id = location + "(lat: " + str(lat) + ", lon: " + str(lon) + ", date_from: " + str(date_from) + ")"
        target_id = "\"" + location + "\"";
        
        target=Target(target_id, lat,lon,lat_nw,lon_nw,lat_se,lon_se,location,date_from, date_to, country,is_direct,calibrated,kind,figurative, age_est)

        target_list[target_id] = target

    return target_list
    
def filter_targets_for_not_direct(target_list):
    print("length of original list=",len(target_list))

    filtered_list = {};
    for key, target in target_list.iteritems():
        if target.is_direct == 'Yes':
            filtered_list[key] = target;

    print("length of filtered list=",len(filtered_list))
    
    return filtered_list

def filter_targets_for_not_exact_age(target_list):
    print("length of original list=",len(target_list))

    filtered_list = {};
    for key, target in target_list.iteritems():
        if target.age_estimation.lower()=='exact age':
            filtered_list[key] = target;

    print("length of filtered list=",len(filtered_list))

    return filtered_list

def filter_targets_for_not_figurative(target_list):
    print("length of original list=",len(target_list))
    

    filtered_list = {};
    for key, target in target_list.iteritems():
        if target.figurative == 'Yes':
            filtered_list[key] = target;

    print("length of filtered list=",len(filtered_list))

    return filtered_list

def filter_targets_for_date(target_list, minimum_date, maximum_date):
    filtered_list = {};
    for key, target in target_list.iteritems():
        if target.date_from >= minimum_date and target.date_from <= maximum_date:
            filtered_list[key] = target

    return filtered_list

def filter_targets_for_latitude(target_list, minimum_lat, maximum_lat):
    filtered_list = {};
    for key, target in target_list.iteritems():
        lat = target.orig_lat
        if lat >= minimum_lat and lat <= maximum_lat:
            filtered_list[key] = target;
    return filtered_list


def create_binned_column(dataframe, new_column_name, base_column_name, interval):

    conditions = [];
    choices = [];
    min_val = int(dataframe[base_column_name].min()/interval);
    max_val = int(dataframe[base_column_name].max()/interval) + 1;

    for i in range(min_val, max_val):
        lower_bound = i*interval;
        upper_bound = i*interval + (interval-1);
        condition = ((dataframe[base_column_name] >= lower_bound) & (dataframe[base_column_name] <= upper_bound));
        choice = str(lower_bound) + "-" + str(upper_bound);
        conditions.append(condition);
        choices.append(choice);

    dataframe[new_column_name] = np.select(conditions, choices);

    return dataframe

def generate_merged_dataframe(base_path, directory, dataframe, globals_dataframe, save_processed_targets):

    temp_globals_df = globals_dataframe.copy();
    temp_sites_df = dataframe.copy();
    temp_sites_df = temp_sites_df[temp_sites_df.type == 's'];

     # DELETE COLUMNS
    temp_sites_df.drop(["target_id","sites_growth_coefficient", "distance", "is_dir", "is_exact", "pseudo_type", "rank", "target_date_from", "target_date_to", "target_lat", "target_location", "target_lon","type"], axis=1, inplace = True)

    # is_site
    temp_globals_df['is_site'] = 0;    
    temp_sites_df['is_site'] = 1;

    # MERGE
    to_concat = [temp_globals_df, temp_sites_df]
    merged_df = pd.concat(to_concat,sort=True);
  #  merged_df.drop(["Unnamed: 0"], axis = 1, inplace = True)
    

    if save_processed_targets:
        processed_targets_dir = os.path.join(base_path, "processed_targets")
        merged_df_filename = os.path.join(processed_targets_dir, directory + "_merged_df.csv") 
        if not os.path.exists(processed_targets_dir):
            os.makedirs(processed_targets_dir)
        merged_df.to_csv(merged_df_filename, sep=";", index=False)
        print("Saved merged dataframe: " + merged_df_filename);
        
    return merged_df