import numpy as np

def write_label(a_file, label):
    for i in range(0, len(label)+8):
        a_file.write('*')
    a_file.write('\n')
    a_file.write(label + "\n")
    for i in range(0, len(label)+8):
        a_file.write('*')
    a_file.write('\n')

def write_headers(a_file, headers, delimiter):
    for i in range(0, len(headers)):
        a_file.write(str(headers[i]))
        if i != len(headers)-1:
            a_file.write(delimiter)
    a_file.write('\n')
    
def write_list(a_file, label, list_to_write):
    write_label(a_file, label);
    for i in list_to_write:
        a_file.write(str(i) + "\n")

def write_table(a_file, label, headers, values, delimiter):
    write_label(a_file, label)
    write_headers(a_file, headers, delimiter)
    
    rows = len(values)
    columns = len(values[0])
    for j in range(0, columns):
        for i in range(0, rows):
            a_file.write(str(values[i][j]))
            if i != rows-1:
                a_file.write(delimiter)
        a_file.write('\n')

def write_target(a_file, target, date_window):
    a_file.write('"'+target.location+'"'+';')
    a_file.write(str(int(target.cluster_id))+';')
    a_file.write(str(float(target.orig_lat))+';')
    a_file.write(str(float(target.orig_lon))+';')
    a_file.write(str(int(target.date_from))+';')
    a_file.write(str(int(target.date_to)))
    a_file.write('\n')


def write_parameters(a_file, parameters_filename, parameters, keys):
    a_file.write("parameters_filename; " + parameters_filename + "\n")
    for key in keys:
        a_file.write(key + "; " + str(parameters[key]) + "\n");

def write_target_table(a_file, dataframe, time_window):

    cluster_headers=["Name of Site", "Latitude", "Longitude", "TargetDateFrom", "TargetDateTo", "Direct", "Exact", "Population density target"]
    write_headers(a_file, cluster_headers, ';')

    ################################
    # Group by cluster id and type #
    # Modified to report medians instead of means
    ################################
    # - aggregate by mean
    #new_data = dataframe.groupby(['cluster_id', 'pseudo_type']).mean().reset_index()
    new_data = dataframe.groupby(['target_id', 'pseudo_type']).mean().reset_index()
    temp_types = new_data['pseudo_type'].values
    target_ids = new_data.target_id.unique();
    # print("ClusterID")
    # print(cluster_ids)

    for target_id in target_ids:
        target_df = dataframe[dataframe.target_id == target_id] 
        # print(target_df)
        location = target_df['target_location'].values[0];
        latitude = target_df['target_lat'].values[0];
        longitude = target_df['target_lon'].values[0];
        date_from = target_df['target_date_from'].values[0];
        date_to = target_df['target_date_to'].values[0];

        direct = 'not direct';
        if target_df['is_dir'].values[0]:
            direct = 'direct'

        exact = 'not exact';
        if target_df['is_exact'].values[0]:
            exact = 'exact'

        sample_mean = target_df[target_df.type == 's']['density'].values[0];

        a_file.write("\"" + str(location) + "\";")
        a_file.write(str(latitude) + ";")
        a_file.write(str(longitude) + ";")
        a_file.write(str(date_from) + ";")
        a_file.write(str(date_to) + ";")
        a_file.write(str(direct) + ";")
        a_file.write(str(exact) + ";")
        a_file.write(str(sample_mean) + ";")
        a_file.write("\n")
        
def write_bin_table(a_file, bin_values_df, min_globals):

    write_label(a_file, "Distribution of values for sites and globals")

    columns = ['Bin value', 'Sites', 'Globals', 'Detection Frequency', 'Relative Frequency of Sites', 'Relative Frequency of Globals']
    write_headers(a_file,columns,";")
    bin_array = bin_values_df['bin_array'].values
    sample_counts = bin_values_df['sample_counts'].values
    global_counts = bin_values_df['global_counts'].values
    likelihood_ratios = bin_values_df['likelihood_ratios'].values
    p_samples = bin_values_df['p_samples'].values
    p_globals = bin_values_df['p_globals'].values

    for i in range(0, len(bin_array)):
        a_file.write(str(bin_array[i]) + ';')
        a_file.write(str(sample_counts[i]) + ';')
        a_file.write(str(global_counts[i]) + ';')
        a_file.write('{:.4f}'.format(likelihood_ratios[i]) + ';')
        a_file.write('{:.4f}'.format(p_samples[i]) + ';')
        a_file.write('{:.4f}'.format(p_globals[i]) + ";")
        a_file.write("\n")

def write_analysis(f2, stat_dictionary):

    write_label(f2, "Statistics")

    f2.write('Total sites; '+str(stat_dictionary['total_samples'])+'\n')
    f2.write('Total globals; '+str(stat_dictionary['total_globals'])+'\n\n')

    f2.write('Median density for sites; '+'{:.2f}'.format(stat_dictionary['median_samples'])+'\n')
    f2.write('Median density for globals; '+'{:.2f}'.format(stat_dictionary['median_globals'])+'\n\n')

    f2.write('Mean density for sites; '+'{:.2f}'.format(stat_dictionary['mean_samples'])+'\n')
    f2.write('Mean density for globals; '+'{:.2f}'.format(stat_dictionary['mean_globals'])+'\n\n')

    f2.write('Standard deviation of density for sites; '+'{:.2f}'.format(stat_dictionary['std_samples'])+'\n')
    f2.write('Standard deviation of density for globals; '+'{:.2f}'.format(stat_dictionary['std_globals'])+'\n\n')

def write_likelihood_results(aFile, max_gamma, max_zetta, max_eps, max_likelihood, thresholds,model ):
        write_label(aFile, "Results of max likelihood analysis for "+model+" model")
        if model=='epidemiological' or model=='richard':
            aFile.write('Threshold 0.025;'+ '{:.2f}'.format(thresholds[0])+"\n")
            aFile.write('Threshold 0.25;'+ '{:.2f}'.format(thresholds[1])+"\n")
            aFile.write('Threshold 0.5;'+ '{:.2f}'.format(thresholds[2])+"\n")
            aFile.write('Threshold 0.75;'+ '{:.2f}'.format(thresholds[3])+"\n")
            aFile.write('Threshold 0.975;'+ '{:.2f}'.format(thresholds[4])+"\n")
        
        if model=='epidemiological':
            aFile.write("Max gamma;"+'{:.5f}'.format(max_gamma)+"\n")
            aFile.write("Max eps;"+'{:.5f}'.format(max_eps)+"\n")
#            aFile.write("Max comm="+'{:.2f}'.format(max_comm)+"\n")
        aFile.write("Max zetta;"+'{:.7f}'.format(max_zetta)+"\n")
        aFile.write("Max likelihood;"+'{:.0f}'.format(max_likelihood)+"\n")
        if model=='epidemiological':
            k=3
        else:
            if model=='proportional':
                k=2
            else:
                if model=='constant':
                    k=1                               
        aFile.write("AIC;"+ '{:.2f}'.format(2*k-2*max_likelihood)+"\n")

