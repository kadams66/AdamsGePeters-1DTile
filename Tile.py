import numpy as np
import scipy.sparse as sprs
import warnings
from abc import ABC, abstractmethod

##############################################################################################

class Reactions(ABC):
    def __init__(self, type: str, n_rxns: int, n_ads: int) -> None:
        """
        type : (str) label of type of system/available reactions
        n_rxns : (int) the number of possible reactions in the system
        n_ads : (int) the number of adsorbates (or number of states a site can have - 1) in the system
        Abstract base class for all reaction objects, see subclass docstrings for descriptions of hte rate_consts they expect
        """
        self.type = type #string describing the reactions object type
        self.n_rxns = n_rxns #number of possible reactions
        self.n_ads = n_ads #number of adsorbates

    def get_n_rxns(self) -> int:
        """returns the number of possible reactions"""
        return self.n_rxns
    
    def get_n_ads(self) -> int:
        """returns the number of adsorbates"""
        return self.n_ads

    @abstractmethod
    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        """
        state : (str) current microstate in base (n_ads + 1)
        l : (int) length of Brickwork tile
        d : (int) offset of tiles for periodic boundary
        returns 
        (ndarray) how many of each reaction can occur for the given state
        """
        pass
    
    @abstractmethod
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        """
        state : (str) current microstate in base (n_ads + 1)
        l : (int) length of Brickwork tile
        d : (int) offset of tiles for periodic boundary
        rate_consts : (tuple) list of rate constants (and other relevant parameters) for each of the reactions
        returns 
        (ndarray, ndarray) 
        1) the rates of each possible reaction for the given state, as well as 
        2) the state they go to
        """
        pass

##############################################################################################

class Brickwork():
    def __init__(self, l: int, d: int, reacts: Reactions) -> None:
        """
        l : (int) length of Brickwork tile
        d : (int) offset of tiles for periodic boundary
        reacts : (Reactions) reactions object detailing the reactions for this tile
        """
        self.l = l
        self.d = d
        self.reacts = reacts
        self.n_ads = self.reacts.get_n_ads()
        self.n_states = int((self.n_ads + 1)**l) #number of unique microstates in the tile

    def get_vectors(self) -> tuple[np.ndarray]:
        """
        returns
        1) coverage vectors(columns) for each state (row), and 
        2) the vectors numbering how many possible of each reaction (columns) can occur from each state (row)
        """
        rxn_vecs = np.zeros([self.n_states, self.reacts.get_n_rxns()], dtype=np.uint8)
        cov_vecs = np.zeros([self.n_states, self.n_ads], dtype = np.uint8)
        for i in range(self.n_states):
            state = np.base_repr(i, self.n_ads + 1) #convert index to base (n_ads + 1) number
            state = '0'*(self.l - len(state)) + state #pad beginning with zeros to make length l
            rxn_vecs[i] = self.reacts.get_rxns_number(state, self.l, self.d) #get number of each type of reaction leading from state i
            state_arr = np.array(list(state))
            for j in range(self.n_ads):
                cov_vecs[i, j] = (state_arr == str(j + 1)).sum() #count how many instances of each adsorbate are in the state array
        return cov_vecs, rxn_vecs
    
    def get_theta_ss(self, rate_consts: tuple[float]) -> np.ndarray:
        """
        returns the steady state fractions for each microstate at the given rate constants
        rate_consts : (listlike) list of all the rate constants according to the reaction object chosen
        """
        #data for CSC sparse matrix generating 
        data = [] #rate constants
        indices = [] #row indices
        indptr = [] #index pointer, corresponds to when the above two lists should move to the next column

        #loop over states and obtain all rxn rates
        ind = 0
        for i in range(self.n_states):
            indptr.append(ind)
            state = np.base_repr(i, self.n_ads + 1) #convert index to base n_ads number
            state = '0'*(self.l - len(state)) + state #pad beginning with zeros to make length n
            rates, prod = self.reacts.get_rxns_rate(state, self.l, self.d, rate_consts) #off diagaonal rates, all rxns going from state
            rates.append(-sum(rates)); prod.append(i) #diagonal entry, negative sum of all off diagaonal rates
            data += rates; indices += prod
            ind += len(rates) #move forward index pointer
        indptr.append(ind) #add last index pointer to denote end of data

        #create sparse matrix and solve matrix eqn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sprs.SparseEfficiencyWarning) #shut up the sparse efficiency warnings
            A = sprs.csc_array((data, indices, indptr), shape=[self.n_states, self.n_states]).tolil() #generate steady state transition matrix
            A[-1] = 1. #replace bottom row with all ones for normalization eqn (bottom row was a redundant eqn anyway)
            b = np.zeros(self.n_states); b[-1] = 1. #solution vector: all zeros except the last entry is one
            A = A.tocsc()
            try:
                solver = sprs.linalg.factorized(A) #linear solver for Ax = b
            except RuntimeError:
                #matrix is less than full rank
                print('PANIC : Matrix less than full rank, trying iterative solution')
                try:
                    return sprs.linalg.lgmres(A, b)[0]
                except:
                    print('     Iterative solution failed, saving matrix to file')
                    sprs.save_npz('error_mat.npz', A)
                    raise RuntimeError('Matrix equation solution failed')
        return solver(b)

##############################################################################################
#Example reaction systems below

class Square_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        square lattice, single adsorbate adsorption/desorption, dimerization
        rate constants: (kad, kdes, lambda, krxn)
        """
        super().__init__('Square_Dimer_1Ad', 3, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
            else:
                numbs[1] += 1 #desorption of occupied site
                #association with back neighbor, python automatically handles the negative indices as mod l
                if state[i - 1] == '1': numbs[2] += 1 
                if state[i - d] == '1': numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
            else:
                #desorption of occupied site
                n_adj = (state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #combination with back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': #dimer with i - 1
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[i - d] == '1': #dimer with i - d
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Square_Diffuse_noInt_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        square lattice, single adsorbate adsorption/desorption, diffusion (no interactions), dimerization
        rate constants: (kad, kdes, kdiff, lambda, krxn)
        """
        super().__init__('Square_Diffuse_noInt_Dimer_1Ad', 4, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, diffuse, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
                #diffusion from back neighbor
                if state[i - 1] == '1': numbs[2] += 1
                if state[i - d] == '1': numbs[2] += 1
            else:
                numbs[1] += 1 #desorption of occupied site
                #combination with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': numbs[3] += 1 
                else: numbs[2] += 1
                if state[i - d] == '1': numbs[3] += 1
                else: numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, kdiff, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                #diffusion from back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': #diffusion from i - 1
                    rates.append(kdiff)
                    temp = list(state) #copy current state into list
                    temp[i] = '1'; temp[i - 1] = '0' #change current site to one and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[i - d] == '1': #diffusion from i - d
                    rates.append(kdiff)
                    temp = list(state)
                    temp[i] = '1'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))

            else:
                #desorption of occupied site
                n_adj = (state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #combination with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': #dimer with i - 1
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                else: #diffusion to i - 1
                    rates.append(kdiff)
                    temp = list(state)
                    temp[i] = '0'; temp[i - 1] = '1'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                if state[i - d] == '1': #dimer with i - d
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else: #diffusion to i - d
                    rates.append(kdiff)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '1'
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods
    
class Square_Diffuse_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        square lattice, single adsorbate adsorption/desorption, diffusion, dimerization
        rate constants: (kad, kdes, kdiff, lambda, krxn)
        """
        super().__init__('Square_Diffuse_Dimer_1Ad', 4, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, diffuse, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
                #diffusion from back neighbor
                if state[i - 1] == '1': numbs[2] += 1
                if state[i - d] == '1': numbs[2] += 1
            else:
                numbs[1] += 1 #desorption of occupied site
                #combination with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': numbs[3] += 1 
                else: numbs[2] += 1
                if state[i - d] == '1': numbs[3] += 1
                else: numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, kdiff, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                #diffusion from back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': #diffusion from i - 1
                    n_adj = (state[(i - 2) % l] + state[(i - d - 1) % l] + state[(i + d - 1) % l]).count('1')
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) #copy current state into list
                    temp[i] = '1'; temp[i - 1] = '0' #change current site to one and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[i - d] == '1': #diffusion from i - d
                    n_adj = (state[(i - d - 1) % l] + state[(i - d + 1) % l] + state[(i - d - d) % l]).count('1')
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state)
                    temp[i] = '1'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))

            else:
                #desorption of occupied site
                n_adj = (state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #combination with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': #dimer with i - 1
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                else: #diffusion to i - 1
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state)
                    temp[i] = '0'; temp[i - 1] = '1'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                if state[i - d] == '1': #dimer with i - d
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else: #diffusion to i - d
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '1'
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Hex_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        hexagonal lattice, single adsorbate adsorption/desorption, dimerization
        rate constants: (kad, kdes, lambda, krxn)
        """
        super().__init__('Hex_Dimer_1Ad', 3, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
            else:
                numbs[1] += 1 #desorption of occupied site
                #combination with back neighbor, python automatically handles the negative indices as mod n
                if state[(i - 1) % l] == '1': numbs[2] += 1 
                if state[(i - d) % l] == '1': numbs[2] += 1
                if state[(i - d - 1) % l] == '1': numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions\
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
            else:
                #desorption of occupied site
                n_adj = (state[(i - 1) % l] + state[(i + 1) % l] + state[(i - d) % l] + state[(i - d - 1) % l] + state[(i + d) % l] + state[(i + d + 1) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #combination with back neighbor, python automatically handles the negative indices as mod n
                if state[(i - 1) % l] == '1':
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[(i - 1) % l] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[(i - d) % l] == '1':
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                if state[(i - d - 1) % l] == '1':
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d - 1) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Hex_Diffuse_noInt_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        hexagonal lattice, single adsorbate adsorption/desorption, diffusion (no interactions), dimerization
        rate constants: (kad, kdes, kdiff, lambda, krxn)
        """
        super().__init__('Hex_Diffuse_noInt_Dimer_1Ad', 4, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, diffuse, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
                #diffusion from back neighbor
                if state[(i - 1) % l] == '1': numbs[2] += 1
                if state[(i - d) % l] == '1': numbs[2] += 1
                if state[(i - d - 1) % l] == '1': numbs[2] += 1
            else:
                numbs[1] += 1 #desorption of occupied site
                #combination with/diffusion to back neighbor
                if state[(i - 1) % l] == '1': numbs[3] += 1 
                else: numbs[2] += 1
                if state[(i - d) % l] == '1': numbs[3] += 1
                else: numbs[2] += 1
                if state[(i - d - 1) % l] == '1': numbs[3] += 1
                else: numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, kdiff, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                if state[(i - 1) % l] == '1':
                    #diffusion from i - 1
                    rates.append(kdiff)
                    temp = list(state) #copy current state into list
                    temp[i] = '1'; temp[(i - 1) % l] = '0' #change current site to one and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[(i - d) % l] == '1':
                    #diffusion from i - d
                    rates.append(kdiff)
                    temp = list(state) 
                    temp[i] = '1'; temp[(i - d) % l] = '0' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d - 1) % l] == '1':
                    #diffusion from i - d - 1
                    rates.append(kdiff)
                    temp = list(state) 
                    temp[i] = '1'; temp[(i - d - 1) % l] = '0' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
            else:
                #desorption of occupied site
                n_adj = (state[(i - 1) % l] + state[(i + 1) % l] + state[(i - d) % l] + state[(i - d - 1) % l] + state[(i + d) % l] + state[(i + d + 1) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                if state[(i - 1) % l] == '1':
                    #combination with i - 1
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[(i - 1) % l] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                else:
                    #diffusion to i - 1
                    rates.append(kdiff)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - 1) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d) % l] == '1':
                    #combination with i - d
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else:
                    #diffusion to i - d
                    rates.append(kdiff)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - d) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d - 1) % l] == '1':
                    #combination wtih i - d - 1
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d - 1) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else:
                    #diffusion to i - d - 1
                    rates.append(kdiff)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - d - 1) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods
    
class Hex_Diffuse_Dimer_1Ad_Reactions(Reactions):
    def __init__(self) -> None:
        """
        hexagonal lattice, single adsorbate adsorption/desorption, diffusion, dimerization
        rate constants: (kad, kdes, lambda, krxn)
        """
        super().__init__('Hex_Diffuse_Dimer_1Ad', 4, 1)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption, desorption, diffuse, dimer
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site
                #diffusion from back neighbor
                if state[(i - 1) % l] == '1': numbs[2] += 1
                if state[(i - d) % l] == '1': numbs[2] += 1
                if state[(i - d - 1) % l] == '1': numbs[2] += 1
            else:
                numbs[1] += 1 #desorption of occupied site
                #combination with/diffusion to back neighbor
                if state[(i - 1) % l] == '1': numbs[3] += 1 
                else: numbs[2] += 1
                if state[(i - d) % l] == '1': numbs[3] += 1
                else: numbs[2] += 1
                if state[(i - d - 1) % l] == '1': numbs[3] += 1
                else: numbs[2] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kad, kdes, kdiff, lmbda, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site
                rates.append(kad)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                if state[(i - 1) % l] == '1':
                    #diffusion from i - 1
                    n_adj = (state[(i - 2) % l] + state[(i - d - 1) % l] + state[(i - d - 2) % l] + state[(i + d - 1) % l] + state[(i + d) % l]).count('1')
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) #copy current state into list
                    temp[i] = '1'; temp[(i - 1) % l] = '0' #change current site to one and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                if state[(i - d) % l] == '1':
                    #diffusion from i - d
                    n_adj = (state[(i - d - 1) % l] + state[(i - d + 1) % l] + state[(i - d - d) % l] + state[(i - d - d - 1) % l] + state[(i + 1) % l]).count('1')
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) 
                    temp[i] = '1'; temp[(i - d) % l] = '0' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d - 1) % l] == '1':
                    #diffusion from i - d - 1
                    n_adj = (state[(i - d - 2) % l] + state[(i - d) % l] + state[(i - d - 1 - d) % l] + state[(i - d - 1 - d - 1) % l] + state[(i - 1) % l]).count('1')
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) 
                    temp[i] = '1'; temp[(i - d - 1) % l] = '0' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
            else:
                #desorption of occupied site
                n_adj = (state[(i - 1) % l] + state[(i + 1) % l] + state[(i - d) % l] + state[(i - d - 1) % l] + state[(i + d) % l] + state[(i + d + 1) % l]).count('1')
                rates.append(kdes * lmbda**n_adj)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                if state[(i - 1) % l] == '1':
                    #combination with i - 1
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[(i - 1) % l] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then binary integer
                else:
                    #diffusion to i - 1
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - 1) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d) % l] == '1':
                    #combination with i - d
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else:
                    #diffusion to i - d
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - d) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1)) 
                if state[(i - d - 1) % l] == '1':
                    #combination wtih i - d - 1
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[(i - d - 1) % l] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
                else:
                    #diffusion to i - d - 1
                    rates.append(kdiff * lmbda**n_adj)
                    temp = list(state) 
                    temp[i] = '0'; temp[(i - d - 1) % l] = '1' 
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Square_Assoc_2Ad_Reactions(Reactions):
    def __init__(self, ) -> None:
        """
        square lattice, two adsorbates adsorption/desorption, association
        rate constants: (kadA, kdesA, kadB, kdesB, lambdaAA, lambdaBB, lambdaAB, krxn)
        """
        super().__init__('Square_Assoc_2Ad', 5, 2)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption A, desorption A, adsorption B, desorption B, association
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site with A
                numbs[2] += 1 #adsorption of empty site with B
            elif site == '1':
                numbs[1] += 1 #desorption of A from occupied site
                #association with back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '2': numbs[4] += 1 
                if state[i - d] == '2': numbs[4] += 1
            else:
                numbs[3] += 1 #desorption of B from occupied site
                #association with back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': numbs[4] += 1 
                if state[i - d] == '1': numbs[4] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kadA, kdesA, kadB, kdesB, lmbdaAA, lmbdaBB, lmbdaAB, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            if site == '0':
                #adsorption of empty site with A
                rates.append(kadA)
                prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                #adsorption of empty site with B
                rates.append(kadB)
                prods.append(int(state[:i] + '2' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '2'
            elif site == '1':
                #desorption of A from occupied site
                adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                n_adj_A = adjs.count('1')
                n_adj_B = adjs.count('2')
                rates.append(kdesA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #association with back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '2':
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                if state[i - d] == '2':
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
            else:
                #desorption of B from occupied site
                adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                n_adj_A = adjs.count('1')
                n_adj_B = adjs.count('2')
                rates.append(kdesB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                #association with back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1':
                    rates.append(krxn)
                    temp = list(state) #copy current state into list
                    temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                    prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                if state[i - d] == '1':
                    rates.append(krxn)
                    temp = list(state)
                    temp[i] = '0'; temp[i - d] = '0'
                    prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Square_Diffuse_noInt_Assoc_2Ad_Reactions(Reactions):
    def __init__(self, ) -> None:
        """
        square lattice, two adsorbates adsorption/desorption, diffusion (no interactions), association
        rate constants: (kadA, kdesA, kdiffA, kadB, kdesB, kdiffB, lambdaAA, lambdaBB, lambdaAB, krxn)
        """
        super().__init__('Square_Diffuse_noInt_Assoc_2Ad', 7, 2)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption A, desorption A, diffuse A, adsorption B, desorption B, diffuse B, association
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site with A
                numbs[3] += 1 #adsorption of empty site with B
                if state[i - 1] == '1': numbs[2] += 1 #diffusion A from i-1
                elif state[i - 1] == '2': numbs[5] += 1 #diffusion B from i-1
                if state[i - d] == '1': numbs[2] += 1 #diffusion A from i-d
                elif state[i - d] == '2': numbs[5] += 1 #diffusion B from i-d
            elif site == '1':
                numbs[1] += 1 #desorption of A from occupied site
                #association with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '2': numbs[6] += 1
                elif state[i - 1] == '0': numbs[2] += 1
                if state[i - d] == '2': numbs[6] += 1
                elif state[i - d] == '0': numbs[2] += 1
            else:
                numbs[4] += 1 #desorption of B from occupied site
                #association with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': numbs[6] += 1 
                elif state[i - 1] == '0': numbs[5] += 1
                if state[i - d] == '1': numbs[6] += 1
                elif state[i - d] == '0': numbs[5] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kadA, kdesA, kdiffA, kadB, kdesB, kdiffB, lmbdaAA, lmbdaBB, lmbdaAB, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            match site:
                case '0':
                    #adsorption of empty site with A
                    rates.append(kadA)
                    prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                    #adsorption of empty site with B
                    rates.append(kadB)
                    prods.append(int(state[:i] + '2' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '2'
                    #diffusion to empty site
                    match state[i - 1]:
                        case '1': #diffusion of A
                            rates.append(kdiffA)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - 1] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            rates.append(kdiffB)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - 1] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                    match state[i - d]:
                        case '1': #diffusion of A
                            rates.append(kdiffA)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - d] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            rates.append(kdiffB)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - d] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                case '1':
                    #desorption of A from occupied site
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    rates.append(kdesA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                    prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                    #association with back neighbor, python automatically handles the negative indices as mod n
                    match state[i - 1]:
                        case '2':
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0':
                            rates.append(kdiffA)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                    match state[i - d]:
                        case '2':
                            rates.append(krxn)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '0':
                            rates.append(kdiffA)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                case '2':
                    #desorption of B from occupied site
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    rates.append(kdesB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                    prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                    #association with back neighbor, python automatically handles the negative indices as mod n
                    match state[i - 1]:
                        case '1':
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0':
                            rates.append(kdiffB)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
                    match state[i - d]:
                        case '1':
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0':
                            rates.append(kdiffB)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods
    
class Square_Diffuse_Assoc_2Ad_Reactions(Reactions):
    def __init__(self, ) -> None:
        """
        square lattice, two adsorbates adsorption/desorption, diffusions, association
        rate constants: (kadA, kdesA, kdiffA, kadB, kdesB, kdiffB, lambdaAA, lambdaBB, lambdaAB, krxn)
        """
        super().__init__('Square_Diffuse_Assoc_2Ad', 7, 2)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption A, desorption A, diffuse A, adsorption B, desorption B, diffuse B, association
        for i, site in enumerate(state):
            if site == '0':
                numbs[0] += 1 #adsorption of empty site with A
                numbs[3] += 1 #adsorption of empty site with B
                if state[i - 1] == '1': numbs[2] += 1 #diffusion A from i-1
                elif state[i - 1] == '2': numbs[5] += 1 #diffusion B from i-1
                if state[i - d] == '1': numbs[2] += 1 #diffusion A from i-d
                elif state[i - d] == '2': numbs[5] += 1 #diffusion B from i-d
            elif site == '1':
                numbs[1] += 1 #desorption of A from occupied site
                #association with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '2': numbs[6] += 1
                elif state[i - 1] == '0': numbs[2] += 1
                if state[i - d] == '2': numbs[6] += 1
                elif state[i - d] == '0': numbs[2] += 1
            else:
                numbs[4] += 1 #desorption of B from occupied site
                #association with/diffusion to back neighbor, python automatically handles the negative indices as mod n
                if state[i - 1] == '1': numbs[6] += 1 
                elif state[i - 1] == '0': numbs[5] += 1
                if state[i - d] == '1': numbs[6] += 1
                elif state[i - d] == '0': numbs[5] += 1
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kadA, kdesA, kdiffA, kadB, kdesB, kdiffB, lmbdaAA, lmbdaBB, lmbdaAB, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            match site:
                case '0':
                    #adsorption of empty site with A
                    rates.append(kadA)
                    prods.append(int(state[:i] + '1' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '1'
                    #adsorption of empty site with B
                    rates.append(kadB)
                    prods.append(int(state[:i] + '2' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '2'
                    #diffusion to empty site
                    match state[i - 1]:
                        case '1': #diffusion of A
                            adjs = state[i - 2] + state[i - d - 1] + state[(i + d - 1) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - 1] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            adjs = state[i - 2] + state[i - d - 1] + state[(i + d - 1) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - 1] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                    match state[i - d]:
                        case '1': #diffusion of A
                            adjs = state[i - d - 1] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - d] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            adjs = state[i - d - 1] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - d] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                case '1':
                    #desorption of A from occupied site
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    rates.append(kdesA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                    prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                    #association with back neighbor, python automatically handles the negative indices as mod n
                    match state[i - 1]:
                        case '2': #combo with i - 1
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0': #diffusion to i - 1
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                    match state[i - d]:
                        case '2': #combo with i - d
                            rates.append(krxn)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '0': #diffusion to i - d
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                case '2':
                    #desorption of B from occupied site
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    rates.append(kdesB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                    prods.append(int(state[:i] + '0' + state[i+1:], self.n_ads + 1)) #changes character at position `i` to '0'
                    #association with back neighbor, python automatically handles the negative indices as mod n
                    match state[i - 1]:
                        case '1': #combo with i - 1
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0': #diffusion to i - 1
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
                    match state[i - d]:
                        case '1': #combo with i - d
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '0': #diffusion to i - d
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
        return rates, prods

class Square_Diffuse_Assoc_Dual_Reactions(Reactions):
    def __init__(self, ) -> None:
        """
        square lattice, two dimer adsorbates dual adsorption/desorption, diffusion, association
        rate constants: (kadA2, kdesA2, kdiffA, kadB2, kdesB2, kdiffB, lambdaAA, lambdaBB, lambdaAB, krxn)
        """
        super().__init__('Square_Diffuse_Combo_2Ad', 7, 2)

    def get_rxns_number(self, state: str, l: int, d: int) -> np.ndarray:
        numbs = np.zeros(self.n_rxns, dtype=np.uint8)
        #adsorption A2, desorption A2, diffuse A, adsorption B2, desorption B2, diffuse B, association
        for i, site in enumerate(state):
            if site == '0':
                if state[i - 1] == '0': 
                    numbs[0] += 1 #adsorption A2 with i-1
                    numbs[3] += 1 #adsorption B2 with i-1
                elif state[i - 1] == '1': numbs[2] += 1 #diffusion A from i-1
                elif state[i - 1] == '2': numbs[5] += 1 #diffusion B from i-1
                if state[i - d] == '0': 
                    numbs[0] += 1 #adsorption A2 with i-d
                    numbs[3] += 1 #adsorption B2 with i-d
                elif state[i - d] == '1': numbs[2] += 1 #diffusion A from i-d
                elif state[i - d] == '2': numbs[5] += 1 #diffusion B from i-d
            elif site == '1':
                if state[i - 1] == '0': numbs[2] += 1 #diffusion A to i-1
                elif state[i - 1] == '1': numbs[1] += 1 #desorption A2 with i-1
                elif state[i - 1] == '2': numbs[6] += 1 #association AB with i-1
                if state[i - d] == '0': numbs[2] += 1 #diffusion A to i-d
                elif state[i - d] == '1': numbs[6] += 1 #desorption A2 with i-d
                elif state[i - d] == '2': numbs[6] += 1 #association AB with i-d
            else:
                if state[i - 1] == '0': numbs[5] += 1 #diffusion B to i-1
                elif state[i - 1] == '1': numbs[6] += 1 #association AB with i-1
                elif state[i - 1] == '2': numbs[3] += 1 #desorption B2 with i-1
                if state[i - d] == '0': numbs[5] += 1 #diffusion B to i-d
                elif state[i - d] == '1': numbs[6] += 1 #association AB with i-d
                elif state[i - d] == '2': numbs[3] += 1 #desorption B2 with i-d
        return numbs
    
    def get_rxns_rate(self, state: str, l: int, d: int, rate_consts: tuple[float]) -> tuple[np.ndarray]:
        (kadA2, kdesA2, kdiffA, kadB2, kdesB2, kdiffB, lmbdaAA, lmbdaBB, lmbdaAB, krxn) = rate_consts
        rates = [] #reaction rates
        prods = [] #products of reactions
        for i, site in enumerate(state):
            match site:
                case '0':
                    match state[i - 1]:
                        case '0': #adsorption of A2 and B2
                            #adsorption of dual empty site with A2
                            rates.append(kadA2)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - 1] = '1' #change current site to A and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                            #adsorption of dual empty site with B2
                            rates.append(kadB2)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - 1] = '2' #change current site to B and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '1': #diffusion of A
                            adjs = state[(i - 1 - 1) % l] + state[(i - 1 - d) % l] + state[(i - 1 + d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - 1] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            adjs = state[(i - 1 - 1) % l] + state[(i - 1 - d) % l] + state[(i - 1 + d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - 1] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                    match state[i - d]:
                        case '0': #adsorption of A2 and B2
                            #adsorption of dual empty site with A2
                            rates.append(kadA2)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - d] = '1' #change current site to A and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                            #adsorption of dual empty site with B2
                            rates.append(kadB2)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - d] = '2' #change current site to B and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '1': #diffusion of A
                            adjs = state[i - d - 1] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '1'; temp[i - d] = '0' #change current site to A and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #diffusion of B
                            adjs = state[i - d - 1] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A = adjs.count('1')
                            n_adj_B = adjs.count('2')
                            rates.append(kdiffA * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state) #copy current state into list
                            temp[i] = '2'; temp[i - d] = '0' #change current site to B and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                case '1':
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    match state[i - 1]:
                        case '0': #diffusion A to i - 1
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '1': #desorption A2 with i - 1
                            adjs2 = state[(i - 1 - 1) % l] + state[(i - 1 - d) % l] + state[(i - 1 + d) % l]
                            n_adj_A_des = adjs2.count('1') + n_adj_A - 1 #minus 1 to account for the interal interaction
                            n_adj_B_des = adjs2.count('2') + n_adj_B
                            rates.append(kdesA2 * lmbdaAA**n_adj_A_des * lmbdaAB**n_adj_B_des)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #assoc AB with i - 1
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                    match state[i - d]:
                        case '0': #diffusion to i - d
                            rates.append(kdiffA * lmbdaAA**n_adj_A * lmbdaAB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '1' #change current site to zero and adjacent site to A
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '1': #desorption A2 with i - d
                            adjs2 = state[(i - d - 1) % l] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A_des = adjs2.count('1') + n_adj_A - 1 #minus 1 to account for the interal interaction
                            n_adj_B_des = adjs2.count('2') + n_adj_B
                            rates.append(kdesA2 * lmbdaAA**n_adj_A_des * lmbdaAB**n_adj_B_des)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #assoc with i - d
                            rates.append(krxn)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1))
                case '2':
                    adjs = state[i - 1] + state[(i + 1) % l] + state[i - d] + state[(i + d) % l]
                    n_adj_A = adjs.count('1')
                    n_adj_B = adjs.count('2')
                    match state[i - 1]:
                        case '0': #diffusion to i - 1
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - 1] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '1': #assoc with i - 1
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #desorption B2 with i - 1
                            adjs2 = state[(i - 1 - 1) % l] + state[(i - 1 - d) % l] + state[(i - 1 + d) % l]
                            n_adj_A_des = adjs2.count('1') + n_adj_A 
                            n_adj_B_des = adjs2.count('2') + n_adj_B - 1 #minus 1 to account for the interal interaction
                            rates.append(kdesB2 * lmbdaAB**n_adj_A_des * lmbdaBB**n_adj_B_des)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - 1] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                    match state[i - d]:
                        case '0': #diffusion to i - d
                            rates.append(kdiffB * lmbdaAB**n_adj_A * lmbdaBB**n_adj_B)
                            temp = list(state)
                            temp[i] = '0'; temp[i - d] = '2' #change current site to zero and adjacent site to B
                            prods.append(int(''.join(temp), self.n_ads + 1))
                        case '1': #assoc with i - d
                            rates.append(krxn)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
                        case '2': #desorption B2 with i - 1
                            adjs2 = state[(i - d - 1) % l] + state[(i - d + 1) % l] + state[(i - d - d) % l]
                            n_adj_A_des = adjs2.count('1') + n_adj_A 
                            n_adj_B_des = adjs2.count('2') + n_adj_B - 1 #minus 1 to account for the interal interaction
                            rates.append(kdesB2 * lmbdaAB**n_adj_A_des * lmbdaBB**n_adj_B_des)
                            temp = list(state) #copy current state into list
                            temp[i] = '0'; temp[i - d] = '0' #change current site and adjacent site to zero
                            prods.append(int(''.join(temp), self.n_ads + 1)) #convert back to string then ternary integer
        return rates, prods
