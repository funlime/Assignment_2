

import time
import pickle
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar
import copy
from copy import deepcopy

import matplotlib.pyplot as plt   
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({"axes.grid" : True, "grid.color": "black", "grid.alpha":"0.25", "grid.linestyle": "--"})
plt.rcParams.update({'font.size': 14})


from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


#-------------------------------------------------------------------------------------------------------------
#                       New obj funct
#-------------------------------------------------------------------------------------------------------------
def u_gov(x, model):

    """ss discounted sum of utility  given government level"""

    par = model.par
    ss = model.ss
    path = model.path

    par.G_ = x
    model.find_ss(do_print=False)

    model.test_path(in_place=False) 
    model.compute_jacs(do_print=False)
    model.find_transition_path(shocks=[],do_print=False)

    U =np.sum([par.beta**t * np.sum(path.u[t]*path.D[t]/np.sum(path.D[t])) for t in range(par.T)])
    
    return - U

def obj_gov(x, model):

    """Objective function with government as the choice variable"""

    return - u_gov(x, model)

# New objetctiv function 
def u_gov_chi_old(x, model):
    """ss discounted sum of utility  given government level and transfers"""

    par = model.par
    ss = model.ss
    path = model.path

    par.G_ = x[0]
    par.Chi_ = x[1]

    model.find_ss(do_print=False)

    #model.find_transition_path(shocks=[],do_print=False)
    model.test_path(in_place=False) 
    model.compute_jacs(do_print=False)
    model.find_transition_path(shocks=[],do_print=False)

    U =np.sum([par.beta**t * np.sum(path.u[t]*path.D[t]/np.sum(path.D[t])) for t in range(par.T)])

    return  U    
    #return  U

def obj_gov_chi(x, model):
    """Objective function with government and chi as the choice variable"""
    return - u_gov_chi(x, model)


# New objetctiv function 
def u_gov_chi(x, model):
    par = model.par
    ss = model.ss
    path = model.path

    """ss discounted sum of utility  given government level"""
    par = model.par
    ss = model.ss
    path = model.path

    model.par.G_ = x[0]
    model.par.Chi_ = x[1]

    model.find_ss(do_print=False)

    #model.find_transition_path(shocks=[],do_print=False)
    model.test_path(in_place=False) 
    model.compute_jacs(do_print=False)
    model.find_transition_path(shocks=[],do_print=False)

    U =np.sum([par.beta**t * np.sum(path.u[t]*path.D[t]/np.sum(path.D[t])) for t in range(par.T)])
    
    return - U


#-------------------------------------------------------------------------------------------------------------
#                       Tables / extra calculations
#-------------------------------------------------------------------------------------------------------------


def table_ss(model):

    par = model.par
    ss = model.ss
    path = model.path

    data = {varname: [f'{model.ss.__dict__[varname]:.3f}'] for varname in model.varlist}
    df = pd.DataFrame(data).T  # Transpose to get variables as rows
    df.columns = ['Value']
    df.index.name = 'Variable'

    return df

def table_extra(x, model):
    # Existing code for setting up parameters and calculating values
    par = model.par
    ss = model.ss
    path = model.path

    # i. Input 
    par.G_ = x[0]
    par.Chi_ = x[1]
    U = u_gov_chi(x, model)

    # iv. Preparing data for the DataFrame
    data = {
        'Chi_val': ss.Chi,
        'Gov_opt': ss.G,
        'Gamma_Y': ss.Gamma_Y,
        'U_opt': -U,
        'Y_G_relatio': ss.G/ss.Y
    }

    # Create DataFrame from the data
    df = pd.DataFrame(data, index=['Value']).T  # Transpose to get variables as rows
    df.index.name = 'Variable'

    # Format the numbers to three decimal places
    df['Value'] = df['Value'].apply(lambda x: f'{x:.3f}')

    return df





def table_extra_new(x, model):

    par = model.par
    ss = model.ss
    path = model.path

    # i. Input 
    par.G_ = x[0]
    par.Chi_ = x[1]

    U = u_gov_chi(x, model)


    Chi_val = ss.Chi
    Gov_val = ss.G
    U_val = - U
    Y_G_relatio = ss.G/ss.Y
    Gamma_Y = ss.Gamma_Y

    # iv. Dictionary for resultsis (Optimum values )
    df = {
        'Chi_val': Chi_val,
        'Gov_opt': Gov_val,
        'Gamma_Y': Gamma_Y,
        'U_opt': U_val,
        'Y_G_relatio': Y_G_relatio,
    }  

    return df




def calc_chi(chi, model, lists = False):

    par = model.par
    ss = model.ss
    path = model.path

    # i. Input 
    par.Chi_ = chi

    # ii. Minimizing
    result = minimize_scalar(obj_gov, bounds=(0.36, 0.475), method='bounded', args=(model))

    # iii. Rerunning
    obj_gov(result.x, model)

    Chi_val = ss.Chi
    Gov_val = result.x
    U_val = result.fun
    Y_G_relatio = ss.G/ss.Y
    Gamma_Y = ss.Gamma_Y


    # iv. Dictionary for resultsis (Optimum values )
    result_dict = {
        'Chi_val': Chi_val,
        'Gov_opt': Gov_val,
        'Gamma_Y': Gamma_Y,
        'U_opt': U_val,
        'Y_G_relatio': Y_G_relatio,
    }  

    # iv. List of U and Gov values for plotting 
    if lists == True:

        g_values = [0.36, 0.39, 0.4, 0.42, 0.45, 0.475]
        u_values = [u_gov(x, model) for x in g_values]
        
        result_dict['g_values'] = g_values
        result_dict['u_values'] = u_values


    return result_dict







#-------------------------------------------------------------------------------------------------------------
#                       Figures 
#-------------------------------------------------------------------------------------------------------------

def plot_policy(model, new_output):
    ss = model.ss
    par = model.par
    path = model.path

    i_fix = 0

    fig = plt.figure(figsize=(18,4),dpi=100)
    a_max = 100

    # a. consumption
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,1)
    ax.set_title(f'consumption')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.c[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.legend(frameon=True)
    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('consumption, $c_t$')

    # b. saving
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,2)
    ax.set_title(f'saving')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.a[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('savings, $a_{t}$')

    # c. labor supply
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,3)
    ax.set_title(f'labor_supply')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.ell[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('labor supply, $n_{t}$')
    fig.tight_layout()

    if new_output == True:
        print('Exporting')
        plt.savefig('figs/policy_functions.png',dpi=200)
    
    else:
        print('Not exporting')
        plt.show()




def plot_cdf(model, new_output):
    ss = model.ss
    par = model.par
    path = model.path

    fig = plt.figure(figsize=(12,4),dpi=100)
    # a. income
    ax = fig.add_subplot(1,2,1)
    ax.set_title('productivity')

    y = np.cumsum(np.sum(ss.D,axis=(0,2)))
    ax.plot(par.z_grid,y/y[-1])

    ax.set_xlabel('productivity, $z_{t}$')
    ax.set_ylabel('CDF')

    # b. assets
    ax = fig.add_subplot(1,2,2)
    ax.set_title('savings')
    y = np.insert(np.cumsum(np.sum(ss.D,axis=(0,1))),0,0.0)
    ax.plot(np.insert(par.a_grid,0,par.a_grid[0]),y/y[-1])
            
    ax.set_xlabel('assets, $a_{t}$')
    ax.set_ylabel('CDF')
    ax.set_xscale('symlog')

    #save figure
    if new_output == True:
        print('Exporting')
        fig.savefig('figs/distribution.png', bbox_inches='tight')
    
    else:
        print('Not exporting')
        plt.show()




def plot_utility(model, new_output):
    ss = model.ss
    par = model.par
    path = model.path

    model.test_path(in_place=True)

    #Ploting the accumulation discounted utility
    par.T = 500
    time_ = [1, 10, 20, 30, 40, 50,  100, 150, 200, 250, 300, 350, 400, 450, 500]


    time_past = []
    disc_utility = []
    for i in time_:
        par.T = i 
        time_past.append(i)
        U =np.sum([par.beta**t * np.sum(path.u[t]*path.D[t]/np.sum(path.D[t])) for t in range(par.T)])
        disc_utility.append(U)

    print(f'Utility: {disc_utility[-0]:.4f}')

    #figure 
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(time_past, disc_utility, label='Discounted Utility')
    ax.set_xlabel('Time')
    ax.set_ylabel('Utility')
    ax.legend()

    # save figure
    if new_output == True:
        print('Exported')
        fig.savefig('figs/disc_utility.png', bbox_inches='tight')

    else:
        print('Not exported')
        plt.show()


# Assuming chi_results is defined and populated as before


def plot_chi(model, chi_results, new_output, chi0 = True):
    ss = model.ss
    par = model.par
    path = model.path

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('')

    opt_gov_values = []
    opt_u_values = []

    if chi0:
        i = 'Model B'
        ax.plot(chi_results[i]['g_values'], chi_results[i]['u_values'])
        ax.scatter(chi_results[i]['Gov_opt'], chi_results[i]['U_opt'], marker='o', label=f'chi = {i}, optimal utility = {chi_results[i]["U_opt"]:.2f}', color='black')
        opt_gov_values.append(chi_results[i]['Gov_opt'])
        opt_u_values.append(chi_results[i]['U_opt'])
    
    else:
        for i in chi_results.keys():
            ax.plot(chi_results[i]['g_values'], chi_results[i]['u_values'])
            ax.scatter(chi_results[i]['Gov_opt'], chi_results[i]['U_opt'], marker='o', label=f'chi = {i}, optimal utility = {chi_results[i]["U_opt"]:.2f}')
            opt_gov_values.append(chi_results[i]['Gov_opt'])
            opt_u_values.append(chi_results[i]['U_opt'])


    # Plotting the line connecting optimal points
    ax.plot(opt_gov_values, opt_u_values, color='grey', linestyle='--', marker='')

    ax.set_xlabel('Government Production')
    ax.set_ylabel('Utility')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if new_output == True:
        if chi0:
            fig.savefig('figs/chi0_gov.png', bbox_inches='tight')
        else: 
            fig.savefig('figs/chi_gov_all.png', bbox_inches='tight')
    else:
        plt.show()




def plot_chi_u(chi_values, opt_u_values, new_output):
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(chi_values, opt_u_values )
    ax.set_ylabel('Utility')
    ax.set_xlabel('Transfers ($\chi$)')
    ax.axvline(x=0, color='grey', linestyle='--')  # 'r' is for red color, '--' for dashed line
    if new_output:
        fig.savefig('figs/chi_utility.png')
    else:
        fig.show
        print('Not exporting')

