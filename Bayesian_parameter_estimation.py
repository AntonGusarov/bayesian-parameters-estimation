# =============================================================================================
# "Parameter estimation for a toy model of an effective field theory"
# ---------------------------------------------------------------------------------------------
#  Authors:
#  Anton Gusarov, gusarov@kth.se
# ---------------------------------------------------------------------------------------------
#  Bayesian linear regression:
#  estimate the vector of LEC (low-energy constants) for the EFT (effective field theory)
# =============================================================================================

#%config InlineBackend.figure_format ='retina'  # high-quality plots for retina screens
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import emcee
import corner
import scipy
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------------------------
# All the functions defined:
# ---------------------------------------------------------------------------------------------

def load_dataset_D1():
    '''Loads the training dataset D1 from the referenced paper'''
    data = np.loadtxt('D1_c_5.dat')
    x, d, yerr = data[:,0], data[:,1], data[:,2]
    #Returns data points as x,d and absolute error values as yerr
    return x, d, yerr


def generate_new_data(n, scale):
    '''Generates a custom train dataset with variable size and error'''
    #Calls number of data points as n, calls precision as scale
    np.random.seed(42)  # for reproducibility
    x = (1/np.pi) * np.random.rand(n) #Interval [1, 1/pi]
    err = scale * np.random.randn(n) #Absolute error
    yerr = np.abs(err)
    y =  (1/2 + np.tan(np.pi/2*x))**2 + err
    #Returns data points as x,d and absolute error values as yerr
    return x, y, yerr


def log_prior_uniform(*args):
    #*args pass LEC values
    '''Calculates a uniform (aka non-informative) log prior P for the regression coefficients'''
    if np.all(np.absolute(args) < 1000):  # to reproduce the paper's plot, thershold increased up to 1000 instead of 100
        return 0
    #returns logarithmic flat prior
    return -np.inf


def log_prior_gaussian(*args, a_fix=5): 
    '''Calculates Gaussian log prior P for the regression coefficients'''
    #*args pass LEC values
    model_params = np.array(args)
    return np.log(multivariate_normal.pdf(model_params, cov=np.eye(model_params.size)*(a_fix**2)))


def log_likelihood(x, d, yerr, *args):
    '''Calculates Gaussian log likelihood P'''
    #Calls data set as x, d, yerr, *args pass LEC values
    model_params = np.array(args)
    vandermx = np.vander(x, model_params.size, increasing=True)  # Vandermonde matrix
    model = model_params.dot(vandermx.T)  # polynomial expansion
    s = yerr**2
    #Returns total chi-squared measure logarithmic likelihood
    return -0.5 * np.sum((d-model)**2/s + np.log(s))  # explicit formula


def log_posterior(model_parameters, x, d, yerr, gaussian_prior):
    '''Posterior distribution'''
    #Calls LEC values as model_parameters, data set as x, d, yerr, condition for prior as boolean gaussian_prior
    if gaussian_prior is True: 
        log_prior = log_prior_gaussian(model_parameters, a_fix=5)
    elif gaussian_prior is False:
        log_prior = log_prior_uniform(model_parameters)

    #Returns logarithmic posterior evaluated either with flat or natural prior
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + log_likelihood(x, d, yerr, model_parameters)


def bayes_fit(k, gaussian_prior, n_warmup=50, n_steps=25_000, plot_traces=True, plot_corner=True, save_plots=False):
    '''Performs the MCMC sampling and returns the chains with optional resulting plots'''
    #Calls model complexity as k, condition for prior as boolean gaussian_prior
    list_of_true_parameters = [0.25, 1.57, 2.47, 1.29, 4.06]
    list_of_labels = ['$a_0$', '$a_1$', '$a_2$', '$a_3$', '$a_4$', '$a_5', '$a_6$']
    labels = list_of_labels[:k+1]
    true_parameters = list_of_true_parameters[:k+1]
    print('True parameters: ', true_parameters)

    # Set sampler parameters:
    n_dim, n_walkers = k+1, 4*(k+1) # number of walkers as is suggested in the paper
    # starting_guesses = np.random.normal(0, 1, (n_walkers, n_dim))  # real-life style: give correct estimates but ugly plots
    starting_guesses = true_parameters[:k+1] + 5e-4*np.random.randn(n_walkers, n_dim)  # publication style

    # Define sampler object:
    print('Gaussian_prior: ', gaussian_prior)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, args=(x, d, yerr, gaussian_prior))

    # Run the sampler:
    # pos, prob, state = sampler.run_mcmc(starting_guesses, n_warmup) # warm-up, save final positions and reset
    sampler.run_mcmc(starting_guesses, n_warmup) # warm-up, save final positions and reset
    sampler.reset()
    sampler.run_mcmc(starting_guesses, n_steps)
    print(f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):0.3f} in {n_walkers*n_steps} steps')

    # Print estimation results (needed also for Table 3):
    samples = sampler.flatchain
    parameters_sigma = list(np.std(samples, axis=0))
    parameters_median = list(np.median(samples, axis=0))
    print(f'k_max = {k}')
    for i in range(0,len(parameters_median)):
        print(f'a_{i} = {parameters_median[i]:.2f} +/- {parameters_sigma[i]:.2f}')

    # Plot MCMC traces:
    if plot_traces is True:
        fig_traces, axes = plt.subplots(k+1, figsize=(10, 7))
        for i in range(n_dim):
            ax = axes[i]
            ax.plot(samples[:, i], 'royalblue', alpha=0.9)
            ax.set_xlim(0, 5000)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.05, 0.5)
            axes[-1].set_xlabel('step number', fontsize=12)
        if save_plots is True:
            plt.savefig('fig/plot_mcmc_traces.jpg', transparent='False', dpi=840)
            plt.show(fig_traces)

    # Corner-plot - marginal pdfs and projections:
    if plot_corner is True:
        fig_corner = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], truths=true_parameters,
                                   show_titles=True, title_kwargs={'fontsize': 14})
        plt.rcParams.update({'font.size': 14})
        if save_plots is True:
            plt.savefig('fig/plot_corner.jpg', transparent='False', dpi=840)
        plt.show(fig_corner)
    #Returns MCMC-derived LECS
    return samples


def bayes_predict(k, samples, x_predict_range):
    '''Calculate predictions of unobserved data using MCMC-sapled posterior joint pdf'''
    #Calls model complexity as k, MCMC-derived LECs as samples, x values used for x_predict_range
    x_matrix = np.vander(x_predict_range, k+1, increasing=True)
    predictions_ensemble = samples[:,:k+1].dot(x_matrix.T)  # samples[:,:k+1] useful when k ~= k_max

    predictions_quant = np.percentile(predictions_ensemble, [16, 50, 84], axis=0)  # +-sigma and median
    predictions_minus_sigma = predictions_quant[0,:]
    predictions_median      = predictions_quant[1,:]
    predictions_plus_sigma  = predictions_quant[2,:]
    #Returns model predicted f_a(x) values predictions_median and error predictions_minus_sigma, predictions_plus_sigma
    return predictions_minus_sigma, predictions_median, predictions_plus_sigma


def comparison_plot(x, d, yerr, samples, predictions_minus_sigma, predictions_median, predictions_plus_sigma,
                    plot_predict_ensemble=False, save_plots=False):
    '''Plot for data, truth, and predictions'''
    #Calls data set as x, d, yerr, MCMC-derived LECS as samples, model predicted f_a(x) values
    #and error as predictions_media and predictions_minus_sigma, predictions_plus_sigma
    
    # Set 'Times' font for LaTeX in plot annotation:
    # from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Times']})
    # rc('text', usetex=True)
    # plt.rcParams.update({'font.size': 12})

    fig = plt.figure(1, figsize=(4.9, 4.4))
    ax = plt.subplot(111)
    x_lin = np.linspace(0, 1, 100)  # space of predictor (x) 

    # Plot ensemble of predictions:
    if plot_predict_ensemble is True:
        for i in range(0, samples.shape[0], 500):
            poly_model = np.poly1d(samples[i,::-1])
            plt.plot(x_lin, poly_model(x_lin), color='crimson', linewidth=1, alpha=0.01)
    else:
    # Plot posterior predictions +- 1*sigma:
        plt.plot(x_predict_range, predictions_minus_sigma, '-r', linewidth=1)
        plt.plot(x_predict_range, predictions_median, ':r', linewidth=2, label=(f'$k = {k},$' + '$\ k_{max} = $' + f'${k}$'))
        plt.plot(x_predict_range, predictions_plus_sigma, '-r', linewidth=1)
        plt.fill_between(x_predict_range, predictions_minus_sigma, predictions_plus_sigma, color='salmon', alpha=0.4)

    # Plot the true function:
    true_function = lambda x: (1/2 + np.tan(np.pi/2*x))**2  # as suggested in the paper
    plt.plot(x_lin, true_function(x_lin), '-b', linewidth=2.5, label='True function')

    # Plot the observed (train) data:
    plt.errorbar(x, d, yerr=yerr, fmt='ok', mec='k', mfc='g', capsize=3, label='Data')

    # Styling the figure:
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$g(x)$', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.xlim(0, 0.4)
    plt.ylim(0, 1.7)
    plt.tick_params(direction='in', grid_alpha=0.2)
    ax.yaxis.set_ticks([0.4, 0.8, 1.2, 1.6])
    ax.yaxis.set_ticklabels([0.4, 0.8, 1.2, 1.6])
    ax.xaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.xaxis.set_ticklabels([0.0, 0.1, 0.2, 0.3, 0.4])
    plt.grid(True)

    if save_plots is True:
        plt.savefig(f'fig/Plot_predictions_k={k}.jpg', transparent='False', dpi=840)

    plt.show(fig)
    return None


def CHI_dof(samples, k_trunc):
    '''Calculates chi^2/dof as a measure of model fit'''
    #calls MCMC-derived LECS as samples, calls model complexity as k_trunc
    predictions = bayes_predict(k_trunc, samples, x)[1]
    dof = (len(x) - (k_trunc+1))  # dof = num_obs - num_params
    #Returns chi-square test of independence
    return np.sum(((d-predictions)/yerr)**2)/dof


# ---------------------------------------------------------------------------------------------
# Execute:
# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Load the train (observed) data:
    x, d, yerr = load_dataset_D1()  # dataset provided along with the paper
    # x, d, yerr = generate_new_data(3, 0.1)  # our syntetic dataset

    # 2. Setup the model:
    k = 4 # model complexity, (k_max in paper's notation)
    k_trunc = 4  # k in paper's notation
    assert (k_trunc <= k), "Model order k is larger than k_max"
    gaussian_prior = False  # False -> uniform_prior

    # 3. Fit the model:
    samples = bayes_fit(k, gaussian_prior, n_warmup=50, n_steps=25_000, plot_traces=False, plot_corner=True, save_plots=True)

    # 4. Predict:
    x_predict_range = np.linspace(start=0, stop=0.4, num=100)  # specify predictors 
    predictions_minus_sigma, predictions_median, predictions_plus_sigma = bayes_predict(k, samples, x_predict_range)

    # 5. Plot predictions with CI, dataset, and ground truth:
    comparison_plot(x, d, yerr, samples, predictions_minus_sigma, predictions_median, predictions_plus_sigma,
                    plot_predict_ensemble=False, save_plots=False)

    # 6. Calulate chi^2/dof:
    print(f'chi^2/dof = {CHI_dof(samples, k_trunc):.1f}')  
