import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def auto_fit(data_to_fit, hist=False, pp=False, dist_names=None):

    y = data_to_fit.dropna().to_numpy()
    size = len(y)
    sc = StandardScaler() 
    yy = y.reshape(-1, 1)
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()
    del yy

    if dist_names is None:
        dist_names = ['beta',
                      'expon',
                      'gamma',
                      'lognorm',
                      'norm',
                      'pearson3',
                      'weibull_min', 
                      'weibull_max']

    # Set up empty lists to store results
    chi_square = []
    p_values = []

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0,100,50)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)    

        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)

    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)

    # Report results

    print('\nDistributions sorted by goodness of fit:')
    print('----------------------------------------')
    print(results)
    
    if hist:
        fitted_histogram(y, results)
        
    if pp:
        pp_qq_plots(y, y_std, dist_names)
        
def fitted_histogram(y, results):    
    size = len(y)
    x = np.arange(len(y))


    # Divide the observed data into 100 bins for plotting (this can be changed)
    number_of_bins = 100
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)

    # Create the plot
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')

    # Get the top three distributions from the previous phase
    number_distributions_to_plot = 5
    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    # Create an empty list to stroe fitted distribution parameters
    parameters = []

    # Loop through the distributions ot get line fit and paraemters

    for dist_name in dist_names:
        # Set up distribution and store distribution paraemters
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)

        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf

        # Add the line to the plot
        plt.plot(pdf_fitted, label=dist_name)

        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(0,np.percentile(y,99))

    # Add legend and display plot

    plt.legend()
    plt.show()

    # Store distribution paraemters in a dataframe (this could also be saved)
    dist_parameters = pd.DataFrame()
    dist_parameters['Distribution'] = (
            results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    # Print parameter results
    print ('\nDistribution parameters:')
    print ('------------------------')

    for index, row in dist_parameters.iterrows():
        print ('\nDistribution:', row[0])
        print ('Parameters:', row[1] )
        
def pp_qq_plots(y, y_std, dist_names):
    ## qq and pp plots
    data = y_std.copy()
    data.sort()
    size = len(y)
    x = np.arange(len(y))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Loop through selected distributions (as previously selected)

    for distribution in dist_names:
        # Set up distribution
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        # Get random numbers from distribution
        norm = dist.rvs(*param[0:-2],loc=param[-2], scale=param[-1],size = size)
        norm.sort()

        # Create figure
        fig = plt.figure(figsize=(8,5)) 

        # qq plot
        ax1 = fig.add_subplot(121) # Grid of 2x2, this is suplot 1
        ax1.plot(norm,data,"o")
        min_value = np.floor(min(min(norm),min(data)))
        max_value = np.ceil(max(max(norm),max(data)))
        ax1.plot([min_value,max_value],[min_value,max_value],'r--')
        ax1.set_xlim(min_value,max_value)
        ax1.set_xlabel('Theoretical quantiles')
        ax1.set_ylabel('Observed quantiles')
        title = 'qq plot for ' + distribution +' distribution'
        ax1.set_title(title)

        # pp plot
        ax2 = fig.add_subplot(122)

        # Calculate cumulative distributions
        bins = np.percentile(norm,range(0,101))
        data_counts, bins = np.histogram(data,bins)
        norm_counts, bins = np.histogram(norm,bins)
        cum_data = np.cumsum(data_counts)
        cum_norm = np.cumsum(norm_counts)
        cum_data = cum_data / max(cum_data)
        cum_norm = cum_norm / max(cum_norm)

        # plot
        ax2.plot(cum_norm,cum_data,"o")
        min_value = np.floor(min(min(cum_norm),min(cum_data)))
        max_value = np.ceil(max(max(cum_norm),max(cum_data)))
        ax2.plot([min_value,max_value],[min_value,max_value],'r--')
        ax2.set_xlim(min_value,max_value)
        ax2.set_xlabel('Theoretical cumulative distribution')
        ax2.set_ylabel('Observed cumulative distribution')
        title = 'pp plot for ' + distribution +' distribution'
        ax2.set_title(title)

        # Display plot    
        plt.tight_layout(pad=4)
        plt.show()