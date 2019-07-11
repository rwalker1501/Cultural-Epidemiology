*************************
Epidemiology of culture data analysis v.1.0
By Richard Walker & Camille Ruiz
*************************

*****************
Purpose
*****************
This repository contains the code and data necessary to reproduce the results contained in the paper, "Epidemiology of human culture: parietal rock art requires high population density" by Richard Walker, Anders Eriksson, Camille Ruiz, Taylor Howard, and Francesco Casalegno.

The parameters for each analysis are contained in an "experiment parameters" file. By modifying these parameters users can run new analyses without modifying the code.

The code provided here is designed to analyse the relationship between parietal rock art and estimates of climate and genetics derived estimates of population density from  Eriksson 2012 and Timmermann 2016 (see references section of the paper).

With minor modifications the same code can be used to analyse  other categories of artifact, and other population estimates

*******************
Usage
*******************

To use the code and data, clone the repository to a local directory.

The program can be used in "script" or "menu" mode.

-----
Script mode
-----

The script contained in the program fig_script.py reproduces the full set of results contained in the paper (plus additional supporting results)

To launch the script, navigate to the working directory containing the code, open a terminal and type

>python fig_script.py

-----
Menu mode
-----
The script main_menu allows the user to load the experiment parameter file for a specific analysis, modify the parameter values, save the parameter values in a new experiment parameter file, and run the corresponding analysis.

To launch the menu system, navigate to the working directory containing the code, open a terminal and type

>python main_menu.py


In script and in menu mode the results of the analysis are saved in a subdirectory of the "Results" directory, with a name defined in the experiment parameters file (see below)

************************
Input Data
************************

The repository contains all the data necessary to reproduce the analyses reported in the paper.

Each analysis uses the following data files

1)  Target file: a .csv file in the "Target" directory containing a description of each rock art site, used for a specific analysis.  For each site, the target provides the following information

- Name
- Latitude
- Longitude
- Earliest age: the earliest date at which art is reported for the site - measured in years BP (years before 1950)
- Latest age: the latest date at which art is reported for the site - measured in years BP
- Modern country: the modern country where the site is located
- Direct/Indirect: whether the date was obtained by direct sampling of the art  or indirectly (e.g. by dating of deposits overlaying the art, dating of other materials found in the same strata etc.)
- Exact date: whether the date provided is an exact date, a minimum age or a maximum age
- Calibrate: whether RC dates have been calibrated
- Kind: the kind of art (painting, drawing, engraving etc.)
- Figurative: whether the art is figurative (this is a comment field not used in the quantitative analysis)

2) Population data file:

This file contains estimates of human population density for a three dimensional array of latitudes, longitudes, and dates.

Two files are currently available.

Eriksson.npz: This file contains the population estimates used in Eriksson 2012 plus modified estimates for the Americas from Raghaven, 2015

Timmermann.npz: This file contains the population estimates used in the early exit scenario reported in Timmermann 2016.

The files contain a compressed representation of four numpy arrays

- dens_txt: the human population density in a hexagonal cell with latitude, longitude and date determined by the corresponding data in the lats, longs and ts files
- lats_txt: latitudes - one per cell
_ lons_txt: longitudes - one per cell
_ ts_txt:	dates - one per cell

Units of density and time are specified in a separate .info file (see below)

3) Population info file

Each population file is associated with a meta-data json file with the following fields

time_multiplier: a coefficient scaling the time units used in the file to years BP
density_multiplier: a coefficient scaling the human population density units used in the file to individuals/100 km2
time_window: in the analysis population densities at a given latitude are aggregated across a time window whose width is specified by this parameter
ascending_time: if ascending_time is True the dates in ts_txt are sorted from the most recent date to the latest date, if False, they are sorted in the inverse order

4) Experiment parameters file

This json file contains all the parameters necessary to run an experiment. The file can be modified by hand, or from the menu. The parameters are defined as follows:

---
Population data:
---
The name of the population data file

---
Globals type:
---
The class of globals used in the analysis:

Options

All:  All cells in the population file
No equatorials: All cells in the population excluding cells between 20°N and 10°S
Australia: All cells within the territory of modern Australia
France_Spain: All cells within the territory of modern France and Spain

---
Target file
---
Name of the target file

---
Results file
----
Name of the results directory

---
Bin size
---

Size of the population density bins shown in the results file.
Note: the likelihood analysis uses smaller hard-coded bins

---
Max Population
---

The highest population density (individuals/100km2) shown on the output graphs and the results file

---
Max for uninhabited
---

The maximum population density (individuals/100km2) for “uninhabited cells”. Uninhabited cells are excluded from the analysis.

---
Max date
---

The oldest date considered in the analysis

---
Min date
---

The most recent date considered in the analysis

---
Max latitude
---

The Northern-most latitude considered in the analysis

---
Min latitude
---

The Southern-most latitude considered in the analysis

---
High_res
---

Options:

true: parameters for the likelihood analysis are sampled at high resolution. This setting is mandatory for analysis

false: parameters for the likelihood analysis are sampled at low resolution. This setting is used for rapid testing of new data, parameter values, and software. The results should not be used for analysis.

---
gamma_start
---

The lowest value of gamma considered in the likelihood analysis

---
gamma_end
---

The highest value of gamma considered in the likelihood analysis

---
zetta_start
---

The lowest value of zetta considered in the likelihood analysis

---
zetta_end
---

The highest value of zetta considered in the likelihood analysis

---
eps_start
---

The lowest value of epsilon considered in the likelihood analysis

---
eps_end
---

The highest value of epsilon considered in the likelihood analysis

---
y_acc_start
---

The lowest detection rate considered in the likelihood analysis

---
y_acc_end
---

The highest detection rate considered in the likelihood analysis

---
Remove_not_direct_targets
---

Options

true: sites whose dates were obtained by indirect methods are excluded from the analysis

false:  sites whose dates were obtained by indirect methods are included in the analysis

---
Remove_not_exact_age_targets
---

Options

true: sites where no exact age is available (e.g. targets with only a minimum or a maximum age) are removed from the analysis

false:  sites where no exact age is available (e.g. targets with only a minimum or a maximum age) are included in the analysis

---
Remove_not_figurative_targets
---

Options:

true: sites with no figurative art (e.g. only geometric patterns) are removed from the analysis

false: sites with no figurative art (e.g. only geometric patterns) are included in the analysis

---
Save_processed_targets
---

Options

true: the program saves a .csv file suitable for analysis with external software. The file contains the estimate population density of every inhabited cell for every time period, together with its status as a site/non-site. The file is typically large (~1 gbyte) and its generation may require several minutes

false: the.csv file is not generated

************************
Output
************************

Each experiment produces the following output, which is stored in a subdirectory of the results directory

---
<name> + "constant_zetta"
---
A graph showing the posterior distribution of the zetta parameter for the constant model in the likelihood analysis

---
<name>+ "epidemiological_eps"
---
A graph showing the posterior distribution of the eps parameter for the epidemiological model in the likelihood analysis

---
<name>+ "epidemiological_gamma"
---
A graph showing the posterior distribution of the gamma parameter for the epidemiological model in the likelihood analysis

---
<name>+ "epidemiological_zetta"
---
A graph showing the posterior distribution of the zetta parameter for the epidemiological model in the likelihood analysis

---
<name> + "fit to constant model"
---
A graph showing showing the fit of the constant model to the data

---
<name> + "fit to epidemiological model"
---
A graph showing showing the fit of the epidemiological model to the data

----
<name> + "map"
----
A world map showing the locations of the sites and globals used in the analysis

---
<name> + "fit to proportional model"
---
A graph showing showing the fit of the proportional model to the data

---
<name> + proportional_zeta.png
---
A graph showing the posterior distribution of the zetta parameter for the proportional model in the likelihood analysis

---
<name>+relative_frequencies_cumulative.png
----
A graph showing showing the cumulative relative frequencies of sites and globals at different population densities

---
<name>+relative_frequencies.png
----
A graph showing showing the absolute relative frequencies of sites and globals at different population densities

---
Results
---
A .csv file (separator=";") showing the quantitative results of the analysis. The file contains the following information

- The date and time of the analysis
- The name of the experiment parameters file
- The values of each of the experiment parameters
- A list of sites excluded from the analysis (because of user options, because the inferred population density is <=max_for_uninhabited, because the population_data_file contains no datapoint corresponding to the latitude and longitude of the site)
- A list of all site included in the analysis including the attributes of each site, as described in the target file
- Data on the number of sites and globals for each population density bin
     * Density
     * Number of sites;
     * Number of globals
     * Site Frequency (N. sites/N. globals)
     * Relative frequency of sites (actual frequency/mean frequency);
     * Relative Frequency of globals (actual frequency/mean frequency)
- Total number of sites
- Total number of globals
- Median population density for sites
- Median population density for globals
- Mean population density for sites
- Mean population density for globals
- Standard deviation of density for sites
- Standard deviation of density for globals
- Results of the maximum likelihood estimation for the epidemiological model
    * Confidence intervals for the inferred threshold
    * Maximum likelihood estimate of the gamma parameter
    * Maximum likelihood estimate of the epsilon parameter
    * Maximum likelihood estimate of the zetta parameter
    * Maximum likelihood
    * AIC (Akaike Information Criterion)
- Results of the maximum likelihood estimation for the proportional model
    * Maximum likelihood estimate of the zetta parameter
    * Maximum likelihood
    * AIC (Akaike Information Criterion)
- Results of the maximum likelihood estimation for the constant model
    * Maximum likelihood estimate of the zetta parameter (the constant)
    * Maximum likelihood
    * AIC (Akaike Information Criterion)
- Bayes factor: epidemiological vs. proportional model
- Bayes factor: epidemiological vs. constant model

---
Saved data files
---

To save site and globals data for analysis by external programs, set the ""save_processed_targets" parameter in the experiment_parameters file to "true".

With this setting the analysis generates a .csv file with the name <name>+ "_merged_df.csv". The file is stored in the "processed targets" directory.

Note: do not use the <name>+"_dataframe.csv" file. This is used for internal purposes

For each cell, at each date considered in the analysis, the "merged_df" file provides the following information

- Estimated human population density for the cell (individuals/100km2)
- Latitude
- Longitude
- Period: date in years BP
- Is_Sample (1 if the cell contains a site, 0 if it does not contain a site

---
Other files
---

In addition to the files just described, the "processed targets", and "globals" directories contain a number of additional






















