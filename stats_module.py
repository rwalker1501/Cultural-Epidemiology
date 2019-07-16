from __future__ import division #This means division always gives a floating result
import plot_module as plm
import numpy as np
import pandas as pd;
import sys, pickle;
from scipy.stats import linregress, ks_2samp,poisson;
from math import *
from scipy.interpolate import interp1d
import itertools, pickle, gc;



def compute_likelihood_model(directory,results_path, population_data,merged_dataframe,model, parameters):
 
 # This function computes the likelihood of a model and the most likely values for the model parameters. Data is stored in results path
 # The procedure can compute the likelihood of three different classes of model: the epidemiological model, a proportional model where the infected population is proportional
 # to population density and a constant model where the expected size of the infected population is constant.The input data is in the merged data_frame (Note for Camille: this is the data we gave Eriksson - should be possible to simplify)
 # When the low_res parameter is set to true, the system produces low_res graphs. Used for system testing and exploratory testing
 # gamma is the subpopulation extinction rate
 # Zetta is the base probability that a  territorial unit) contains at least one site
 # Eps is an error - means that there is a positive probability that a site is present even in a territory with below threshold population
 
    if parameters['high_resolution']:
        res_gamma = 101
        res_zetta = 101
        res_eps = 101
    else:
        res_gamma = 24
        res_zetta = 11
        res_eps = 11

    gamma_v=np.linspace(parameters['gamma_start'], parameters['gamma_end'],num=res_gamma)
    zetta_v=np.exp(np.linspace(log(parameters['zetta_start']),log(parameters['zetta_end']),num=res_zetta,endpoint=False)) 


    if model=='constant':
        eps_v=np.array([1])
    else:
        eps_v=np.linspace(parameters['eps_start'],parameters['eps_end'],num=res_eps,endpoint=False)
    rho_bins=np.linspace(0,33,num=300,endpoint=False)
    rho_bins_4_python=np.append(rho_bins,33)
    # When we show the actual frequencies in Figure 1 - we use smaller bins - with a larger number of samples per bin. This makes the graph easier to read
    # bin_boundaries2_4_python contains the second type of bis
    bin_width=2 
    bin_boundaries2_4_python=np.linspace(0,33,num=16,endpoint=False) 
    bin_boundaries2_4_python=np.append(bin_boundaries2_4_python,33)
    rho_bins2=bin_boundaries2_4_python[0:len(bin_boundaries2_4_python)-1]+bin_width/2 
    # Count the number of samples and controls in each bin
    samples_counts=np.histogram(merged_dataframe['density'][merged_dataframe.is_sample==1],bins=rho_bins_4_python)[0] 
    controls_counts=np.histogram(merged_dataframe['density'][merged_dataframe.is_sample==0],bins=rho_bins_4_python) [0]
    # Repeat for the larger bins
    control_counts2=np.histogram(merged_dataframe['density'][merged_dataframe.is_sample==0],bins=bin_boundaries2_4_python)[0]
    sample_counts2=np.histogram(merged_dataframe['density'][merged_dataframe.is_sample==1],bins=bin_boundaries2_4_python)[0]
    # Compute total number of controls and samples
    n_controls=np.sum(controls_counts)  
    n_samples=np.sum(samples_counts) 
    # Avoid underflow in calculations
    l_shift=n_samples*(log(float(n_samples)/float(n_controls))-1)
    # Computes size of parameter ranges according to range actually chosen - could be cleaned up
    n_gamma=len(gamma_v)
    n_eps=len(eps_v)
    n_zetta=len(zetta_v)
    #  Kills gamma loop for constant and proportional models
    if model=='constant' or model=='proportional':
        n_gamma=1
        # Set up a (population data specific) range of possible values for the likelihood of a given set of observations
    acc_likelihoods=np.linspace(parameters["y_acc_start"],parameters["y_acc_end"],num=2001) 
    #  Set up an array representing the accumulated likelihood of a given set of sample and control counts 
    #  Across all possible values of the parameters
    acc=np.zeros((len(acc_likelihoods),len(rho_bins)))
    lnL=np.zeros((n_gamma,n_eps,n_zetta))
    sqrt_rho_bins=np.sqrt(rho_bins) #These are the values we are computing - rhobins_4_python are intervals for histogram only. In original program were inside loop. Have moved it outside
    max_LL=-float('inf') 
    bin_zeros=np.zeros(len(sqrt_rho_bins))
    # Scan all possible values of the parameters
    for i_gamma in range (0,n_gamma):
        print 'Percentage completed=', i_gamma/float(n_gamma)
        my_gamma=float(gamma_v[i_gamma]) 
        # Compute the predicted size of the infected population  as a proportion of population (p_infected) for all possible values of rho_bins, given the value of gamma. Guarantee it is always 0 or greater
        if model=='epidemiological':
            p_infected=np.maximum(bin_zeros,1-my_gamma/sqrt_rho_bins)  
        if model=='proportional' or model=='constant':
            p_infected=rho_bins/max(rho_bins)
        for i_zetta in range(0,n_zetta):
            for i_eps in range (0, n_eps):
                 my_zetta=zetta_v[i_zetta]
                 my_eps=eps_v[i_eps]
                 # Predicts the probability of finding at least one site in a territory for each of the population densities in rho_bins. Make sure value is never too small
                 if model=='proportional':
                     p_predicted=compute_proportional_model(p_infected,my_zetta,my_eps)
                 else:
                     if model=='constant':
                         p_predicted=compute_constant_model(p_infected,my_zetta,my_eps)
                     else:
                         if model=='epidemiological':
                             p_predicted=compute_epidemiological_model(p_infected,rho_bins,my_zetta,my_eps)
                             # Computes the log likelihood of obtaining the OBSERVED number of samples at a given value of rho_bins, given the predicted number of samples
                  
                 log_samples=np.dot(samples_counts,np.log(p_predicted)) 
                 # The same for controls
                 log_controls=np.dot(controls_counts,np.log(1-p_predicted)) 
                 # Computes the log likelihood of a certain number of samples AND a certain number of controls (for a given value of rho_bins)
                 LL=log_samples+log_controls
                 # Finds the parameter values with the maximum likelihood
                 if LL>max_LL:
                     max_LL=LL
                     max_gamma=my_gamma
                     max_zetta=my_zetta
                     max_eps=my_eps
                     max_likelihood=LL
                 if np.isnan(np.min(LL)):
                     print 'LL is nan'
                     print 'nSamples=',n_samples
                     print 'nControls=',n_controls
                     print 'my_gamma=',my_gamma
                     print 'my_zetta=',my_zetta
                     print 'my_eps=',my_eps
                     print 'log_samples',log_samples
                     print 'log_controls', log_controls
                     print 'p_predicted=', p_predicted
                     print 'samples_counts=', samples_counts
                     print 'controls_counts=',controls_counts
                     sys.exit()
                         
                     # Stores the log likelihood in an array indexed the position of the parameter values in the parameter ranges

                 lnL[i_gamma,i_eps,i_zetta]=LL 
                 # Computes the actual likelihood of the observations and applies a left shift to make sure it is not too large (This means values shown are relative only)
                 L=np.exp(LL-l_shift)
                 len_acc_likelihoods=np.array(len(acc_likelihoods))
                 len_acc_likelihoods.fill(len(acc_likelihoods))
                 # Create a one dimensional array of indexes pointing to values in acc_likelihoods (e.g. possible likelihood values) corresponding to different values of pObs,COMPLEX - WOULD BE NICE TO HAVE EASIER APPROACH. 
                 i_acc=np.minimum(len_acc_likelihoods,np.floor(1+p_predicted/acc_likelihoods[1]).astype(int)) #This yields column vector of indexes corresponding to different values of pObs. ector length =401. Maximum value of index =400 (zero based vector).I am keeping it 1-based
                 for i in range(0,len(i_acc)):
                     x_coord=i_acc[i]-1
                     y_coord=i
                     # Accumulate likelihood values (x coord) for a each possible value of rho_bins (y_coord) across all values of the parameters
                     acc[x_coord,y_coord]=acc[x_coord,y_coord]+L

    interpolated_gammas=plm.plot_parameter_values(lnL,gamma_v, zetta_v, eps_v,model,directory,results_path)
    thresholds=interpolated_gammas**2
    opt_threshold=thresholds[2]  #Not sure about this
    plm.plot_maximum_likelihood(acc,rho_bins,rho_bins2,acc_likelihoods, gamma_v, opt_threshold, sample_counts2, control_counts2, model,directory,results_path)
    return(max_gamma, max_zetta, max_eps, max_likelihood,thresholds)
    

def compute_epidemiological_model(p_infected,rho_bins,my_zetta,my_eps):
    p_predicted=np.zeros(len(rho_bins)).astype(float) 
    p_predicted=my_zetta*((1-my_eps)*p_infected*rho_bins)+my_eps
    p_predicted_small=np.zeros(len(p_predicted))
    p_predicted_large=np.zeros(len(p_predicted))
    p_predicted_large.fill(1-0.000000001)
    p_predicted_small.fill(1e-20)
    p_predicted=np.maximum(p_predicted,p_predicted_small)
    p_predicted=np.minimum(p_predicted,p_predicted_large)
    p_predicted=p_predicted.astype(float)
    return(p_predicted)
    
def compute_proportional_model(p_infected, my_zetta,my_eps):
    p_predicted=np.zeros(len(p_infected)).astype(float)
    p_predicted=my_zetta*p_infected
    p_predicted_small=np.zeros(len(p_predicted))
    p_predicted_large=np.zeros(len(p_predicted))
    p_predicted_small.fill(1e-20)
    p_predicted_large.fill(1-0.000000001)
    p_predicted=np.maximum(p_predicted,p_predicted_small)
    p_predicted=np.minimum(p_predicted,p_predicted_large)
    p_predicted=p_predicted.astype(float) #Probably not necessary
    return(p_predicted)

def compute_constant_model(p_infected, my_zetta,my_eps):
    p_predicted=np.zeros(len(p_infected)).astype(float)
    p_predicted.fill(my_zetta)
    p_predicted_small=np.zeros(len(p_predicted))
    p_predicted_large=np.zeros(len(p_predicted))
    p_predicted_small.fill(1e-20)
    p_predicted_large.fill(1-1e-20)
    p_predicted=np.maximum(p_predicted,p_predicted_small)
    p_predicted=np.minimum(p_predicted,p_predicted_large)
    p_predicted=p_predicted.astype(float)
    return(p_predicted)


def process_dataframe(dataframe):

    conditions = [];
    samples_growth_coefficients = []
    valid_ids = []

    target_ids = dataframe.target_id.unique();
    removed_targets = []


    for target_id in target_ids:
        target_df = dataframe[dataframe.target_id==target_id]
        sample_target_df = target_df[target_df.type == 's']
        if np.isnan(sample_target_df['density'].median()):
            print('Removing target: ' + str(target_id));
            removed_targets.append(target_id);
            dataframe = dataframe[dataframe.target_id != target_id];
            continue;

        # extract all periods and all population as arrays from the samples dataframe
        sample_times = sample_target_df['period'].values
        sample_populations = sample_target_df['density'].values

        # compute growth coefficients for samples
        growth_coefficient_samples=compute_growth_coefficient(sample_times, sample_populations)

        valid_ids.append(target_id);
        samples_growth_coefficients.append(growth_coefficient_samples)

    for target_id in valid_ids:
        conditions.append((dataframe['target_id'] == target_id));

    dataframe['samples_growth_coefficient'] = np.select(conditions, samples_growth_coefficients);
    return dataframe, removed_targets;

def compute_growth_coefficient(times, populations):
    if len(times)>=2:
        for i in range(0, len(populations)):
            if np.isnan(populations[i]):
                populations[i] = 0
        slope, intercept, r_value, p_value, std_err = linregress(times, populations)
        return slope 
    else:
        return -1.


def generate_bin_values_dataframe(dataframe, globals_dataframe, bin_size, max_population, minimum_globals):

    # minimum_bin=max_for_uninhabited
    minimum_bin = 0
    bins_to_omit=int(minimum_bin/bin_size)

    ######################
    # Create bin columns #
    ######################
    # creating bins according to density of the row

    # main dataframe
    dataframe['bin_index'] = (dataframe.density/bin_size)-bins_to_omit
    dataframe['bin_index'] = dataframe.bin_index.astype(int)
    dataframe = dataframe[dataframe.bin_index >= 0]
    dataframe['bin'] = dataframe.bin_index*bin_size+minimum_bin

    # globals dataframe
    globals_dataframe['bin_index'] = (globals_dataframe.density/bin_size)-bins_to_omit
    globals_dataframe['bin_index'] = globals_dataframe.bin_index.astype(int)
    #we add bin_size/2 to get midpoint of each bin
    globals_dataframe['bin'] = globals_dataframe.bin_index*bin_size+minimum_bin
    bin_array = []
    sample_counts = []
    global_counts = []
    likelihood_ratios = []
    p_samples = []
    p_globals = []

    ##############
    # Get Totals #
    ##############
    # total samples by summing contributions
    # total globals by counting rows
    total_samples = dataframe[dataframe.type=='s']['density'].count()
    total_globals = globals_dataframe['density'].count()
    
    #########################
    # Loop through each bin . data untrimmed - would be better to filter here#
    #########################
    current_bin = minimum_bin
    while(current_bin < max_population):
        
        bin_array.append(current_bin)
        # sample count: for all samples in the bin, sum all contributions
        samples_dataframe = dataframe[dataframe.type=='s']
        current_sample_count = samples_dataframe[samples_dataframe.bin == current_bin]['density'].count()
        if np.isnan(current_sample_count):
            current_sample_count = 0;
        sample_counts.append(current_sample_count)
        
        # global count: count all globals dataframe rows in the bin
        current_global_count = globals_dataframe[globals_dataframe.bin == current_bin]['density'].count()
        if np.isnan(current_global_count):
            current_global_count = 0;
        global_counts.append(current_global_count)
        
        # likelihood ratio: sample_count/global_count - probably no lomger necessary
        likelihood_ratio = -1
        if(current_global_count != 0):
            likelihood_ratio = float(current_sample_count)/current_global_count
        likelihood_ratios.append(likelihood_ratio)
        
        # p_sample: sample_count/total_samples
        p_sample = -1
        if total_samples > 0:
            p_sample = float(current_sample_count)/total_samples
        p_samples.append(p_sample)
        
        # p_global: global_count/total_globals
        p_global = -1
        if total_globals > 0:
            p_global = float(current_global_count)/total_globals
        p_globals.append(p_global)

        current_bin += bin_size

    df = pd.DataFrame({'bin_array': bin_array, 'sample_counts': sample_counts, 'global_counts': global_counts, 'likelihood_ratios': likelihood_ratios, 'p_samples': p_samples, 'p_globals': p_globals})

    return df;

def generate_statistics(dataframe, globals_dataframe, bin_values_df, minimum_globals):

    trimmed_bin_values_df = bin_values_df[bin_values_df.global_counts > minimum_globals];
    trimmed_bin_values_df['cum_p_samples'] = trimmed_bin_values_df.p_samples.cumsum();
    trimmed_bin_values_df['cum_p_globals'] = trimmed_bin_values_df.p_globals.cumsum();

    
    if len(trimmed_bin_values_df.index) < len(bin_values_df.index)/2:
        return None, None;

    stat_dictionary = {};

    stat_dictionary['trimmed_bin_values_df'] = trimmed_bin_values_df;

    stat_dictionary['total_samples'] = dataframe[dataframe.type=='s']['density'].count()
    stat_dictionary['total_globals'] = globals_dataframe ['density'].count()

    stat_dictionary['median_samples'] = dataframe[dataframe.type=='s']['density'].median()
    stat_dictionary['median_globals'] = globals_dataframe ['density'].median()


    stat_dictionary['mean_samples'] = dataframe[dataframe.type=='s']['density'].mean()
    stat_dictionary['mean_globals'] = globals_dataframe ['density'].mean()

    stat_dictionary['std_samples'] = dataframe[dataframe.type=='s']['density'].std()
    stat_dictionary['std_globals'] = globals_dataframe ['density'].std();


    trimmed_p_samples = trimmed_bin_values_df['p_samples'].values;
    trimmed_p_globals = trimmed_bin_values_df['p_globals'].values;
    return stat_dictionary, trimmed_bin_values_df;
