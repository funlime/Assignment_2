import time
import numpy as np
from scipy import optimize

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):

        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    for i_fix in range(par.Nfix):
        
        # a. raw value
        ell = 1.0
        y = ss.wt*ell*par.z_grid
        c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
        v_a = (1+ss.r)*c**(-par.sigma)

        # b. expectation
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a

def obj_ss(x,model,do_print=False):

    KL = x[0] # Guessing on capital  
    tau =  x[1] # Guessing on tau

    par = model.par
    ss = model.ss

    # Dividing labor types 
  
    # a. firms
    ss.rK = par.alpha*par.Gamma_Y*(KL)**(par.alpha-1)
    ss.w = (1.0-par.alpha)*par.Gamma_Y*(KL)**par.alpha


    # b. Name ---
    ss.r = ss.rK - par.delta
    ss.wt = (1-tau)*ss.w

    # c. government - 
    ss.Gamma_G    = par.Gamma_G_ini    # Should it be a parameter, sinse it is time invariant but it is a shock 
    ss.G        = par.G_
    ss.L_G      = ss.Gamma_G * ss.G #par.L_G_ss
    ss.Chi      = par.Chi_
    ss.S = ss.G # From eq 8 

    
    # d. households HouseholdProblem solved from wt, r and chi  
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)


    # e. market clearing
   
    ss.L = ss.L_hh                     #*** Clearing_L clearer mekanisk  -  Ud fra det her clearer markederne rent mekanisk, er clearing_L overflødig, er den her overflødig? Eller beregner vi den her for at bruge og så det bare pænt at have begge steder
    ss.L_Y = ss.L - ss.L_G
    ss.K = KL*ss.L_Y

    ss.tau = tau

    ss.Y = par.Gamma_Y*ss.K**(par.alpha)*ss.L_Y**(1-par.alpha) # Total privat production
    ss.I = par.delta*ss.K
    ss.A = ss.K                        

    ss.clearing_A = ss.A - ss.A_hh     
    ss.clearing_L = ss.L - ss.L_hh
    ss.clearing_Y = ss.Y - (ss.C_hh + ss.I + ss.G)    
    ss.clearing_G = ss.G + (ss.w*ss.L_G + ss.Chi) - tau * ss.w * ss.L_hh # Gov budget constraint 


    #return np.array([ss.clearing_A])
    return  ss.clearing_A , ss.clearing_G


def find_ss(model,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    KL_min = ((1/par.beta+par.delta-1)/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1)) + 1e-2
    KL_max = (par.delta/(par.alpha*par.Gamma_Y))**(1/(par.alpha-1))-1e-2
    KL_mid = (KL_min+KL_max)/2 # middle point between max values as initial capital labor ratio


    # a. solve for K and L
    initial_guess =  (KL_mid, 0.0) 
#    initial_guess =  np.array([KL_mid])
    
    if do_print: print(f'starting at [{initial_guess[0]:.4f}]')

    res = optimize.root(obj_ss, initial_guess, args=(model,))
    if do_print: 
        print('')
        print(res)
        print('')
    
    # b. final evaluations
    obj_ss(res.x,model)

    # c. show
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.K = :6.3f}')

        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')