"""Convert specific txt files to netcdf format"""

# pylint: disable=R0914
# THIS SEEMS LIKE UP TO DATE VERSION
# from __future__ import division
import numpy as np
import os, gc;
import json
import target_module as tam
import population_data_module as pdm
import stats_module as stm
import plot_module as plm
import write_module as wrm
import pandas as pd
from datetime import datetime
from scipy.stats import binom
pd.options.mode.chained_assignment = None 



#==============================================================================
# 'plt.style.use('ggplot')
#==============================================================================

class MainProgram:


    def __init__(self):
        self.base_path = os.getcwd();
        self.parameters_folder = os.path.join(self.base_path, "experiment_parameters");
        self.targets_folder = os.path.join(self.base_path,"targets");
        self.parameters_filename = "default_experiment_param.txt"
        self.key_order = keys = ["population_data", "globals_type", "target_file", "results_directory", "bin_size", "max_population", "max_for_uninhabited", "max_date","min_date", "max_lat", "min_lat", "high_resolution", "gamma_start", "gamma_end","zetta_start", "zetta_end", "eps_start", "eps_end", "y_acc_start", "y_acc_end", "remove_not_direct_targets", "remove_not_exact_age_targets", "remove_not_figurative_targets", "save_processed_targets", "min_globals","infer_missing_values","phi"];


    ##################
    # Setter Methods #
    ##################


    def set_parameters_filename(self, parameters_filename):
        self.parameters_filename = parameters_filename;


    ##################
    # Getter Methods #
    ##################

    def get_targets_folder(self):
        return self.targets_folder

    def get_parameters_folder(self):
        return self.parameters_folder;

    def get_parameters_filename(self):
        return self.parameters_filename

    def add_population_data(self, name, binary_path, info_path):
        new_population_data = pdm.load_population_data_source(name, binary_path, info_path)
        self.population_data_sources.append(new_population_data)

    #########################
    # Target List Functions #
    #########################


    def save_target_list(self, filename, some_target_list):
        tests_path=os.path.join(self.base_path,"targets")
        tam.save_target_list_to_csv(some_target_list, tests_path, filename)
        self.dataframe = pd.DataFrame()
        self.dataframe_loaded = False;

    #############################
    # Generate Results Function #
    #############################

    def generate_results(self, parameters_filename=None):
        if parameters_filename is None:
            parameters_filename = self.parameters_filename;

        parameters_filepath = os.path.join(self.parameters_folder, parameters_filename);
        parameters = json.load(open(parameters_filepath));

        population_data = pdm.load_population_data_source(self.base_path, parameters['population_data']);

        target_filepath = os.path.join(self.targets_folder, parameters['target_file'])
        target_list = tam.read_target_list_from_csv(target_filepath);
        
        #####################################
        # Create directory and results file #
        #####################################
        results_path = os.path.join(self.base_path, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        directory = parameters["results_directory"];
        new_path = os.path.join(results_path, directory)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        results_filename= os.path.join(new_path, directory + "_results.csv") 

        f2= open(results_filename, 'w')


        ############################
        # Write header information #
        ############################
        wrm.write_label(f2, 'Epidemiology of culture data analysis v.1.1 Copyright (C) 2019,2020  Richard Walker & Camille Ruiz')
        dateTime=str(datetime.now())
        f2.write('Date: '+dateTime)
        f2.write('\n')
        wrm.write_parameters(f2, parameters_filename, parameters, self.key_order)


        ##########################################
        # Process targets and extract dataframes #
        ##########################################
        target_list, targets_dataframe, globals_dataframe = tam.process_targets(self.base_path, population_data, target_list, parameters)
        if targets_dataframe.empty or len(target_list)<10:
            f2.write("Not enough sites in Target Areas")
            f2.close()
            return "Not enough sites in target area"

        print("Processing sites and globals dataframe...")
        targets_dataframe, removed_targets = stm.process_dataframe(targets_dataframe,parameters['infer_missing_values'])
        if parameters['infer_missing_values']:
            n_removed=len(removed_targets)
        else:
            wrm.write_list(f2, "Removed Targets", removed_targets);
            n_removed=0

        print("Saving merged sites and globals dataframes...")
        merged_dataframe = tam.generate_merged_dataframe(self.base_path, directory, targets_dataframe, globals_dataframe, parameters['save_processed_targets']);
       
        ########################################
        # Write filtered clustered target list #
        ########################################
        print("Writing target list...")
        wrm.write_label(f2, "Filtered Target List")
        wrm.write_target_table(f2, targets_dataframe, population_data.time_window)
        
        #################
        # Generate bins #
        #################
        # - extracts bin values
        # - write bin values to file
        print("Writing bins...")
        bin_values_df = stm.generate_bin_values_dataframe(targets_dataframe, globals_dataframe, parameters['bin_size'], parameters['max_population'], parameters['min_globals'],n_removed)
        wrm.write_bin_table(f2, bin_values_df,parameters['min_globals'])


        #################
        # Stat Analysis #
        #################
        # - n, median, mean, std of sites and non_sites
        # - K-S2 test for bin distribution of sites (p_sites) vs. bin distribution of non_sites (p_non_sites)
        print("Calculating statistics...")
        stat_dictionary, trimmed_bin_values_df = stm.generate_statistics(targets_dataframe, globals_dataframe, bin_values_df, parameters['min_globals'])
        
        if stat_dictionary is None:
            f2.write('insufficient non-zero bins for analysis');
            return 'insufficient non-zero bins for analysis';
        
        print("Writing analysis...")
        wrm.write_analysis(f2, stat_dictionary);
        
        ###############
        # Compare likelihoods of epidemiological, proportional and null models 
        ###############
        print("Computing likelihoods Models")
        models=('epidemiological', 'proportional','null')
        max_likelihood=np.zeros(len(models))
        for i in range(0,len(models)):
            print("model= " + models[i])
            max_gamma, max_zetta, max_eps, max_likelihood[i], thresholds=stm.compute_likelihood_model(directory, results_path, population_data,merged_dataframe, models[i], parameters)
            wrm.write_likelihood_results(f2,max_gamma, max_zetta, max_eps, max_likelihood[i], thresholds,models[i],parameters['phi'] );
            gc.collect();
            if i==0:
                opt_threshold=thresholds[2]
        epid_over_proportional=np.exp(max_likelihood[0]-max_likelihood[1])
        epid_over_constant=np.exp(max_likelihood[0]-max_likelihood[2])
        wrm.write_label(f2,'Bayes factors')
        f2.write( 'Bayes factor epidemiological over proportional;'+'{:.3g}'.format(epid_over_proportional)+'\n')
        f2.write( 'Bayes factor epidemiological over null;'+'{:.3g}'.format(epid_over_constant)+'\n')
        
        below_th_sites=merged_dataframe.loc[(merged_dataframe['is_site']==True) & (merged_dataframe['density']<opt_threshold)]
        below_th_non_sites=merged_dataframe[(merged_dataframe['is_site']==False) & (merged_dataframe['density']<opt_threshold)]
        s=below_th_sites['density'].count()
        n_non_sites=stat_dictionary['total_non_sites']
        n_non_sites_below_th=below_th_non_sites['density'].count()
        p_below_th=float(n_non_sites_below_th)/float(n_non_sites)
        n_sites=stat_dictionary['total_sites']
        p= binom.cdf(s, n_sites, p_below_th) 
        wrm.write_label(f2,'Binomial test')
        f2.write('Number of sites with density < threshold: '+'{:.3g}'.format(s)+'\n')
        f2.write('Number of non-sites with density<threshold: '+'{:.3g}'.format(n_non_sites_below_th)+'\n')
        f2.write('Proportion of non-sites below threshold: '+'{:.3g}'.format(p_below_th)+'\n')
        f2.write('Probability of below threshold site number (binomial): '+'{:.3g}'.format(p)+'\n')
        f2.close();
        
        

        
        ###############
        # Plot Graphs #
        ###############

        print("Generating graphs...")
        plm.plot_stat_graphs(stat_dictionary, trimmed_bin_values_df, population_data, parameters['bin_size'], parameters['max_population'], opt_threshold,directory, new_path);

        # plots targets and non_sites on a map
        plm.plot_targets_on_map(targets_dataframe, globals_dataframe, new_path, directory)
        
        
        return max_likelihood
    

def run_experiment(*args):

    mp = MainProgram()
    if len(args) == 0:
        mp.generate_results();
    elif len(args) == 1 and isinstance(args[0], str):
        mp.generate_results(parameters_filename=args[0]);
    else:
        print("Input a single parameter (string).")
    gc.collect();

