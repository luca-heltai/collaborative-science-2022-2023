import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial
from jax import lax

from jax.tree_util import tree_flatten, tree_map
from jax.flatten_util import ravel_pytree
from jax.example_libraries.stax import (serial, Dense, Tanh)

import time

jax.config.update('jax_platform_name', 'cpu')

# DEFINE THE NEURAL NET CLASS
class Wavefunction(object):
    def __init__(self, key, nstates, ndense):
        self.key = key
        self.nstates = nstates
        self.activation = Tanh
        self.ndense = ndense
        self.alpha = 8
        
    def build(self):
        self.psi_a_init, self.psi_a_apply = serial(
            Dense(self.ndense), self.activation,
            Dense(self.ndense), self.activation,
            Dense(self.ndense), self.activation,
            Dense(1),
        )
        self.key, key_input = jax.random.split(self.key)
        in_shape = (-1, self.nstates) 
        psi_a_shape, psi_a_params = self.psi_a_init(key_input, in_shape)
        self.num_psi_a_params = len(psi_a_params)
        
        self.psi_p_init, self.psi_p_apply = serial(
            Dense(self.ndense), self.activation,
            Dense(self.ndense), self.activation,
            Dense(self.ndense), self.activation,
            Dense(1),
        )
        self.key, key_input = jax.random.split(self.key)
        psi_p_shape, psi_p_params = self.psi_p_init(key_input, in_shape)
        self.num_psi_p_params = len(psi_p_params)

        net_params = psi_a_params + psi_p_params
        
        net_params = tree_map(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)
        num_flat_params = flat_net_params.shape[0]

        return net_params, num_flat_params

    # Calculates wf
    @partial(jit, static_argnums=(0,))
    def psi(self, params, ni):
        num_offset_params = 0
        psi_a_params = params[num_offset_params : num_offset_params + self.num_psi_a_params]
        num_offset_params = num_offset_params + self.num_psi_a_params
        psi_p_params = params[num_offset_params : num_offset_params + self.num_psi_p_params]
        
        psiout = jnp.exp(self.alpha * jnp.tanh(self.psi_a_apply(psi_a_params, ni)/self.alpha)) * jnp.tanh(self.psi_p_apply(psi_p_params, ni))
        #psiout = jnp.exp(self.psi_a_apply(psi_a_params, ni)) * jnp.tanh(self.psi_p_apply(psi_p_params, ni))
        
        return jnp.reshape(psiout, ())

    # Batched version
    @partial(jit, static_argnums=(0,))
    def vmap_psi(self, params, ni_batched):
        return vmap(self.psi, in_axes=(None, 0))(params, ni_batched)

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters 

    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters

    @partial(jit, static_argnums=(0,))
    def update_cast(self, params):
        return params.astype(jnp.float64)


# DEFINE THE HAMILTONIAN CLASS
class Hamiltonian(object):
    def __init__(self, npart, nstates, dvec, gmat, wavefunction):
        self.npart = npart
        self.nstates = nstates
        self.wavefunction = wavefunction

        # Initialize 1-body and 2-body potentials
        self.dvec = dvec
        self.gmat = gmat

    @partial(jit, static_argnums=(0,))
    def pot1body(self, ni):
        return 2*jnp.dot(ni, self.dvec)

    @partial(jit, static_argnums=(0,))
    def vmap_1body(self, ni_batched):
        return vmap(self.pot1body, in_axes=0)(ni_batched)

    # 2 body potential
    @partial(jit, static_argnums=(0,))
    def pot2body(self, params, ni):

        cN = self.wavefunction.psi(params, ni)

        def qbody(j, carry):
            aivec, p, ni1, cumul = carry
            q = aivec[j]
            ni2 = ni1.at[q].add(1)

            cNp = self.wavefunction.psi(params, ni2)
            cumul += cNp*self.gmat[p, q]
            return aivec, p, ni1, cumul

        def pbody(i, carry):
            v, ivec, aivec, ni = carry
            p = ivec[i]
            ni1 = ni.at[p].add(-1)
            aivec, p, _, pterm2 = lax.fori_loop(0, self.nstates-self.npart, qbody, (aivec, p, ni1, 0)) # Sum over all unoccupied spins
            v += pterm2
            return v, ivec, aivec, ni

        ivec = jnp.nonzero(ni, size=self.npart)[0] # Array of occupied sites
        nip = 1 - ni
        aivec = jnp.nonzero(nip, size=self.nstates-self.npart)[0] # Array of unoccupied sites

        v, ivec, aivec, ni = lax.fori_loop(0, self.npart, pbody, (0, ivec, aivec, ni))

        return v/cN + jnp.dot(ni, jnp.diag(self.gmat))

    @partial(jit, static_argnums=(0,))
    def vmap_2body(self, params, ni_batched):
        return vmap(self.pot2body, in_axes=(None, 0))(params, ni_batched)

    @partial(jit, static_argnums=(0,))
    def energy(self, params, ni_batched):
        e1 = self.vmap_1body(ni_batched)
        e2 = self.vmap_2body(params, ni_batched)
        en = e1 + e2
        #return ke, pe, en
        return en

# DEFINE THE MONTE CARLO CLASS
class Metropolis(object):
    def __init__(self, npart, nstates, nwalk, neq, nav, nac, nvoid, wavefunction):

        self.npart = npart
        self.nstates = nstates
        self.nwalk = nwalk 
        self.neq = neq
        self.nav = nav
        self.nac = nac
        self.nvoid = nvoid
        self.wavefunction = wavefunction

        # Number of ordered pairs that can be formed starting from ntot elements
        self.npair = int((self.nstates) * ((self.nstates) - 1) / 2)
        k = 0
        ipnp = np.empty(self.npair, dtype=int)
        jpnp = np.empty(self.npair, dtype=int)
        for i in range(0, self.nstates-1):
            for j in range(i+1, self.nstates):
                ipnp[k] = i
                jpnp[k] = j
                k+=1

        self.ip = jnp.array(ipnp)
        self.jp = jnp.array(jpnp)

    # Initializes Fock state (two spins)
    @partial(jit, static_argnums=(0,))
    def nocc_init(self, key):
        key, key_input = random.split(key)
        ni = jnp.zeros((self.nwalk, self.nstates))
        ni = ni.at[:, 0:self.npart].set(1)
        ni = random.permutation(key_input, ni, 1, independent=True)
        return key, ni

    # Exchanges occupation number of two states
    @partial(jit, static_argnums=(0,))
    def nocc_exch(self, ni_o, k):
        # Spin up exchange
        ip = self.ip[k]
        jp = self.jp[k]
        ni_n = ni_o.at[ip].set(ni_o[jp])
        ni_n = ni_n.at[jp].set(ni_o[ip])
        return ni_n

    @partial(jit, static_argnums=(0,))
    def nocc_prop(self, ni_o_batched, k_batched):
        return vmap(self.nocc_exch, in_axes=(0, 0))(ni_o_batched, k_batched)

    # Void steps
    @partial(jit, static_argnums=(0,))
    def step_void(self, key, ni_o, acc, params):
        # Generate all random numbers
        key, key_input = random.split(key)
        k = jax.random.randint(key_input, shape = [self.nvoid, self.nwalk], minval = 0, maxval = self.npair)
        key, key_input = random.split(key)
        unifp = jax.random.uniform(key_input, shape = [self.nvoid, self.nwalk])
        def step(i, carry):
            ni_o, wf_o, acc = carry
            ni_n = self.nocc_prop(ni_o, k[i, :])
            wf_n = self.wavefunction.vmap_psi(params, ni_n)
            prob = ( wf_n / wf_o )**2
            accept = jnp.greater_equal(prob, unifp[i, :])
            ni_o = jnp.where(accept.reshape([self.nwalk,1]), ni_n, ni_o)
            wf_o = jnp.where(accept, wf_n, wf_o)
            acc += jnp.mean(accept.astype('float32'))
            return ni_o, wf_o, acc

        # Initialize log of wavefunction in order to calculate it only once in each loop
        wf_o = self.wavefunction.vmap_psi(params, ni_o)
        ni_o, foo, acc = lax.fori_loop(0, self.nvoid, step, (ni_o, wf_o, acc))
        return key, ni_o, acc

    @partial(jit, static_argnums=(0,))
    def initialize(self, key, nin, params):
        key, ni_o = self.nocc_init(key)
        # Initialization steps
        def initialization(i, carry):
            key, ni_o, acc = carry
            key, ni_o, acc = self.step_void(key, ni_o, acc, params)
            return key, ni_o, acc
        acc = 0
        key, ni_o, acc = lax.fori_loop(0, nin, initialization, (key, ni_o, acc))
        return key, ni_o

    @partial(jit, static_argnums=(0,))
    def walk(self, key, params, ni_o):
        # Equilibrium steps
        def equilibration(i, carry):
            key, ni_o, acc = carry
            key, ni_o, acc = self.step_void(key, ni_o, acc, params)
            return key, ni_o, acc
        acc = 0
        key, ni_o, acc = lax.fori_loop(0, self.neq, equilibration, (key, ni_o, acc))

        ni_stored = jnp.empty((self.nav + self.nac, self.nwalk, self.nstates))  
        # Average steps     
        def average(i, carry):
            key, ni_o, acc, ni_stored = carry
            key, ni_o, acc = self.step_void(key, ni_o, acc, params)
            ni_stored = ni_stored.at[i, :, :].set(ni_o)
            return key, ni_o, acc, ni_stored
        acc = 0
        key, ni_o, acc, ni_stored = lax.fori_loop(0, self.nav+self.nac, average, (key, ni_o, acc, ni_stored))
        acc /= (self.nav+self.nac) * self.nvoid

        return key, acc, ni_stored


# DEFINE THE ESTIMATOR CLASS
class Estimator(object):
    def reset(self):
        self.energy_blk = 0
        self.energy2_blk = 0
        self.weight_blk = 0

    def addval(self, energy):
        self.energy_blk += energy 
        self.energy2_blk += energy**2 
        self.weight_blk += 1

    def average(self):
        self.energy = self.energy_blk / self.weight_blk
        self.energy2 = self.energy2_blk / self.weight_blk

        error = jnp.sqrt((self.energy2 - self.energy**2) / (self.weight_blk-1))
        return error


# RUN CODE
print("Monte Carlo for the nuclear pairing model.")
from jax.config import config
config.update("jax_enable_x64", True)

npart = 5 # Number of pairs
nstates = 10 # Number of energy levels
nin = 30 # Equilibration steps for first initialization of ni
neq = 10 # Equilibration steps
nav = 40 # Averaging steps
nac = 4 # Check steps
nvoid = 200 # Void steps between energy calculations
nwalk = 800 # Quantum Monte Carlo configurations

ndense = 10

seed_net = 73
seed_walk = 103

# Initialize the network with one batch dimension, ndim, and npart
key = random.PRNGKey(seed_net)
wavefunction = Wavefunction(key, nstates, ndense)
params, nparams = wavefunction.build()
print("Number of parameters of the neural net = ", nparams)
    
# Initialize Metropolis sampler 
metropolis = Metropolis(npart, nstates, nwalk, neq, nav, nac, nvoid, wavefunction)

# Initialize Hamiltonian 
gconst = -.6
pvec = jnp.arange(nstates)
gmat = jnp.zeros((nstates, nstates))
for i in range(nstates):
    for j in range(nstates):
        gmat = gmat.at[i, j].set(gconst)
hamiltonian =  Hamiltonian(npart, nstates, pvec, gmat, wavefunction)

# Initialize Estimator
estimator = Estimator()

print("Classes initialized. Initializing MC states...")

# Store the last walker
ni_o = jnp.zeros(shape=[nwalk, nstates])
# Initialize ni for the first time
key = random.PRNGKey(seed_walk)
key, ni_o = metropolis.initialize(key, nin, params)

print("MC states initialized. Performing MC walk...")

# Metropolis energy calculation
twlk_i = time.time()

key, acc, ni_stored = metropolis.walk(key, params, ni_o)
if ni_stored.shape = (nav, nwalk, nstates):
    print("Stored states matrix has the right dimension.")
else:
    raise Exception("Stored states matrix doesn't have the right dimension.")

twlk_f = time.time()
print(f"Walk stored, elapsed time: {(twlk_f - twlk_i):.2f}s. Computing MC energy...")
estimator.reset()
energy_stored = jnp.zeros(shape=[nav, nwalk])

twlk_i = time.time()
for i in range(nav):
    energy = hamiltonian.energy(params, ni_stored[i, :, :])
    energy_stored = energy_stored.at[i,:].set(energy)
    estimator.addval(jnp.mean(energy))
energy.block_until_ready()
twlk_f = time.time()
print(f"Energy computed, elapsed time: {(twlk_f - twlk_i):.2f}s.")

error = estimator.average()
energy = estimator.energy

estimator.reset()

print(f"Energy = {energy:.6f}, err = {error:.6f}.")

