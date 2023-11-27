import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,L_Y,rK,w,Y, Gamma_Y):

    K_lag = lag(ini.K,K)


    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*Gamma_Y*(K_lag/L_Y)**(par.alpha-1.0)

    w[:] = (1.0-par.alpha)*Gamma_Y*(K_lag/L_Y)**par.alpha

    # b. production and investment
    Y[:] = Gamma_Y*K_lag**(par.alpha)*L_Y**(1-par.alpha)


@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K #+ ss.B Hvad er B

    # b. return
    r[:] = rK-par.delta



@nb.njit
def government(par,ini,ss,tau,w,wt, L_G, G, Chi , S, Gamma_G):

    # a. Government production
    S[:] = np.min([G[:], Gamma_G*L_G])


    # b. Transfer
    Chi[:] = ss.Chi
    # c. Taxes
    tau[:] = ss.tau
    wt[:] = (1-tau)*w





@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_hh,Y, w, L, Chi, tau, C_hh,K, L_G ,I,clearing_A,clearing_L,clearing_Y, clearing_G, G, L_Y):

    clearing_A[:] = A-A_hh

    L[:] = L_hh

    clearing_L[:] = L_hh - (L_Y + L_G)
    I[:] = K-(1-par.delta)*lag(ini.K,K)

    clearing_G[:] = G + (w*L_G + Chi) - tau * w * L_hh # Gov budget constraint
    clearing_Y[:] = Y-C_hh-I - G
