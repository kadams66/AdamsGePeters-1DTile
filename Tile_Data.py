import numpy as np
import sys
import os
import shutil
import json
from datetime import datetime, timedelta
from itertools import product
import Tile

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

##################################################################################################

def gen_data(name, n_ads, n_points, l, d, save_loc=None):
    if save_loc == None: save_loc = f'Tile Data/{n_ads}Ad {name}/'
    os.makedirs(save_loc + 'console/', exist_ok=True)
    print(f'l = {l}, d = {d}')

    data_loc = f'kMC Data/{n_ads}Ad {name}/' #kMC data location
    with open(data_loc + 'params.json') as f:
        run_params = json.load(f) #kMC parameters
    ind_run = [] #index of which parameters are varied in the run
    typ_run = [] #the type of variance for the parameters that are varied (log vs linear)
    rxn_params = [] #the reaction parameters, both varied and not, in the order of param_names
    match n_ads:
        case 1: param_names = ['Kad', 'Kdiff', 'eaa', 'Krxn'] #the names/labels of each of the rxn_params, single adsorbate
        case 2: param_names = ['KA', 'KB', 'KdiffA', 'KdiffB', 'Kdes', 'eAA', 'eBB', 'eAB', 'krxn'] #double adsorbate
    #figure out which parameters were varied in the kMC and compile the parameter set
    for i, p in enumerate(param_names):
        if type(run_params[p]) == str:
            ind_run.append(i)
            if run_params[p] == 'geo' or run_params[p] == 'log':
                typ_run.append('geo')
                rxn_params.append(np.geomspace(run_params[p + '_s'], run_params[p + '_e'], n_points))
            elif run_params[p] == 'lin':
                typ_run.append('lin')
                rxn_params.append(np.linspace(run_params[p + '_s'], run_params[p + '_e'], n_points))
            else:
                raise ValueError(p + ' parameter must be either a number or one of ["geo", "log", "lin"]')
        else:
            rxn_params.append([run_params[p]])
    
    match len(ind_run):
        case 1:
            #only 1 parameter is varied, currently only case that mattered
            ind_run = ind_run[0]
            if run_params['latt'] == 'square':
                match n_ads:
                    case 1: rxns = Tile.Square_Diffuse_Dimer_1Ad_Reactions()
                    case 2: rxns = Tile.Square_Diffuse_Assoc_2Ad_Reactions()
            elif run_params['latt'] == 'hex':
                match n_ads:
                    case 1: rxns = Tile.Hex_Diffuse_Dimer_1Ad_Reactions()
                    case 2: rxns = Tile.Hex_Diffuse_Assoc_2Ad_Reactions() #doesn't exist yet
            else: raise ValueError('needs to be square or hex lattice')
            brick = Tile.Brickwork(l, d, rxns) #brickwork tile
            thetas = np.zeros([n_points, brick.n_states + 1])
            thetas[:, 0] = rxn_params[ind_run] #x values for thetas

            ts = []
            told = datetime.now()
            sys_stdout = sys.stdout #save stdout
            with open(save_loc + f'console/l={l}_d={d}.log', 'w') as sys.stdout: #output to log file in folder
                t0 = datetime.now()
                print('start: ', t0)

                for i, ps in enumerate(product(*rxn_params)):
                    print(f'  {i}    ', told)
                    sys.stdout.flush()
                    match n_ads:
                        case 1:
                            (kad, kdiff, eaa, krxn) = ps
                            thetas[i, 1:] = brick.get_theta_ss((kad, 1., kdiff, np.exp(eaa), krxn)) #obtain steady state thetas at the given parameters 
                        case 2:
                            (KA, KB, KdiffA, KdiffB, Kdes, eAA, eBB, eAB, krxn) = ps    
                            thetas[i, 1:] = brick.get_theta_ss((KA, 1.0, KdiffA, KB * Kdes, Kdes, KdiffB * Kdes, np.exp(eAA), np.exp(eBB), np.exp(eAB), krxn))
                    tnew = datetime.now()
                    ts.append(tnew - told)
                    told = tnew
                ave_t = sum(ts, timedelta(0)) / len(ts)
                print(' dt ave:', ave_t)
                std_t = sum([(t / timedelta(microseconds=1) - ave_t / timedelta(microseconds=1))**2 for t in ts]) / len(ts)
                std_t = timedelta(microseconds=np.sqrt(std_t))
                print(' dt std:', std_t)

                t1 = datetime.now()
                print('end:   ', t1)
                print('dt:    ', t1 - t0)
                sys.stdout.flush()
            sys.stdout = sys_stdout #restore stdout to print to console again

            np.savetxt(save_loc + f'l={l}_d={d}.dat', thetas, fmt='%20.8e', header='xs = ' + param_names[ind_run]) #save theta values to disk

def multi_lds(run_name, n_ads, n_points, lds, save_loc=None):
    if save_loc == None: save_loc = f'Tile Data/{n_ads}Ad {run_name}/'
    for ld_index in range(lds.shape[0]):
        l = lds[ld_index, 0]; d = lds[ld_index, 1]
        gen_data(run_name, n_ads, n_points, l, d, save_loc=save_loc)

##################################################################################################

if __name__ == '__main__':
    with open('Paper_params.json') as f:
        data_params = json.load(f)
    run_name = data_params['run_name']
    print(run_name)
    n_ads = data_params['n_ads']
    n_points = data_params['n_points']
    lds = np.array(data_params['lds'])
    try:
        save_loc = data_params['save_loc']
    except KeyError:
        save_loc = f'Tile Data/{n_ads}Ad {run_name}/'
    os.makedirs(save_loc + 'console/', exist_ok=True)
    shutil.copy('Paper_params.json', save_loc)
    multi_lds(run_name, n_ads, n_points, lds, save_loc)
    print('done')