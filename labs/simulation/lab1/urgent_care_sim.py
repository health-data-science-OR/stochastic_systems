'''
Simulation Lab: Urgent Care Call Centre (111) Model
===================================================

This module provides a discrete event simulation (DES) model of an urgent care
call centre in `simpy`. It is the same model we built in the Lab1 Jupyter notebook,
but in an easy to browse format. 

The model is designed to allow you to explore resource allocation (operators and nurses).

The model follows a two-stage triage process:
1. Call Operators: Handle initial triage and assessment.
2. Nurse Callbacks: A proportion of patients (sampled via Bernoulli) receive
   a follow-up call from a clinical nurse.

Key Classes:
-----------
Scenario: 
    Container for simulation configuration, resources, and distributions.
    Handles logging/tracing via the `.log()` method.
Patient: 
    Encapsulates the logic and state of a single patient's journey.
UrgentCareCallCentre: 
    Manages the arrival process and patient entity generation.
Auditor: 
    Monitors resources and collects time-series statistics for reporting.

Example Usage:
-------------
>>> import simpy
>>> import urgent_care_sim as sim
>>> env = simpy.Environment()
>>> # Create a scenario with tracing enabled for debugging
>>> args = sim.Scenario(env)
>>> model = sim.UrgentCareCallCentre(env, args)
>>> env.process(model.arrivals_generator())
>>> env.run(until=sim.RUN_LENGTH)
'''

import itertools
import math
import numpy as np
import pandas as pd
import simpy

# ----------------------------------------------------------------------------
# GLOBAL PARAMETERS (Used by Scenario and helper functions)
# ----------------------------------------------------------------------------
RUN_LENGTH = 1000
N_OPERATORS = 13
N_NURSES = 10

ARRIVAL_RATE = 100
MEAN_IAT = 60 / ARRIVAL_RATE

CALL_LOW = 5
CALL_HIGH = 10
CALL_MODE = 7

# Internal module state for tracing (default to True)
_TRACE = True

# PRNG seeds 
ARRIVAL_SEED = 42
CALL_SEED = 101
CALLBACK_SEED = 1966
NURSE_SEED = 2020

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS & DISTRIBUTIONS
# ----------------------------------------------------------------------------

def set_trace(state: bool):
    '''
    Toggles the simulation trace output on or off.
    
    Usage:
    ------
    Call this before running your simulation to enable/disable printing.
    
    >>> set_trace(True)
    
    Params:
    -------
    state: bool
        True to enable print output, False to disable.
    '''
    global _TRACE
    _TRACE = state
    print(f"Simulation tracing set to: {_TRACE}")

def trace(msg):
    '''
    Prints an event message if tracing is enabled.
    
    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if _TRACE:
        print(msg)

class Exponential():
    '''
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, mean, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the exponential distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.exponential(self.mean, size=size)

class Triangular():
    '''
    Convenience class for the triangular distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, low, mode, high, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        self.mode = mode
        
    def sample(self, size=None):
        return self.rand.triangular(self.low, self.mode, self.high, size=size)

class Bernoulli():
    '''
    Convenience class for the Bernoulli distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, p, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        p: float
            probability of drawing a 1
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.p = p
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.binomial(n=1, p=self.p, size=size)

class Uniform():
    '''
    Convenience class for the Uniform distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, low, high, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        low: float
            lower range of the uniform
            
        high: float
            upper range of the uniform
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.uniform(low=self.low, high=self.high, size=size)

# ----------------------------------------------------------------------------
# CORE MODEL CLASSES
# ----------------------------------------------------------------------------

class Scenario:
    """
    Encapsulates the concept of a Scenario with the urgent care
    call centre simulation model.

    A scenario contains a list of parameters that can be left as defaults or varied.
    """

    def __init__(
        self,
        env: simpy.Environment,
        n_operators: int = N_OPERATORS,
        n_nurses: int = N_NURSES,
        mean_iat: float = MEAN_IAT,
        call_low: float = CALL_LOW,
        call_mode: float = CALL_MODE,
        call_high: float = CALL_HIGH,
        nurse_low: float = 10.0,
        nurse_high: float = 20.0,
        p_callback: float = 0.4,
        main_seed: int = 0,
    ):
        """
        Initialise the scenario with parameters and setup sampling.

        Parameters
        ----------
        env : simpy.Environment
            The SimPy simulation environment.
        n_operators : int, optional
            The number of call operators available, by default N_OPERATORS.
        n_nurses : int, optional
            The number of nurses available for callbacks, by default N_NURSES.
        mean_iat : float, optional
            The mean inter-arrival time (minutes) for incoming calls, by default MEAN_IAT.
        call_low : float, optional
            The minimum duration (minutes) for an operator call triage, by default CALL_LOW.
        call_mode : float, optional
            The most likely duration (minutes) for an operator call triage, by default CALL_MODE.
        call_high : float, optional
            The maximum duration (minutes) for an operator call triage, by default CALL_HIGH.
        nurse_low : float, optional
            The minimum duration (minutes) for a nurse callback, by default 10.0.
        nurse_high : float, optional
            The maximum duration (minutes) for a nurse callback, by default 20.0.
        p_callback : float, optional
            The probability that a patient requires a nurse callback (0.0 - 1.0), by default 0.4.
        main_seed : int, optional
            The initial seed for the pseudo-random number generator streams, by default 0.
        """
        # Model Configuration
        self.env = env
        self.n_operators = n_operators
        self.n_nurses = n_nurses
        self.mean_iat = mean_iat
        self.call_low = call_low
        self.call_mode = call_mode
        self.call_high = call_high
        self.nurse_low = nurse_low
        self.nurse_high = nurse_high
        self.p_callback = p_callback
        
        # Sampling Control
        self.main_seed = main_seed
        self.n_streams = 4  # arrivals, call_duration, callback_decision, nurse_duration
    
        # Resources placeholders (initialised in the model itself)
        self.operators = None 
        self.nurses = None

        # Initialise the PRNG streams
        self.init_sampling()

    def init_sampling(self) -> None:
        """
        Create non-overlapping PRNG streams and assign them to distributions.
        """
        # Create a seed sequence and spawn independent streams
        seed_sequence = np.random.SeedSequence(self.main_seed)
        self.seeds = seed_sequence.spawn(self.n_streams)

        # 1. Call inter-arrival times
        self.arrival_dist = Exponential(
            self.mean_iat, random_seed=self.seeds[0]
        )

        # 2. Duration of operator call triage
        self.call_dist = Triangular(
            self.call_low, self.call_mode, self.call_high,
            random_seed=self.seeds[1]
        )

        # 3. Decision for nurse callback (Bernoulli)
        self.callback_dist = Bernoulli(
            self.p_callback, random_seed=self.seeds[2]
        )

        # 4. Duration of nurse call
        self.nurse_dist = Uniform(
            self.nurse_low, self.nurse_high, 
            random_seed=self.seeds[3]
        )

    def __repr__(self) -> str:
        """
        Returns a detailed string representation of all scenario parameters.
        """
        return (f"Scenario(\n"
                f"  main_seed={self.main_seed},\n"
                f"  n_operators={self.n_operators}, n_nurses={self.n_nurses},\n"
                f"  mean_iat={self.mean_iat:.2f},\n"
                f"  call_dist(low={self.call_low}, mode={self.call_mode}, high={self.call_high}),\n"
                f"  nurse_dist(low={self.nurse_low}, high={self.nurse_high}),\n"
                f"  p_callback={self.p_callback}\n"
                f")")



class Patient:
    '''
    Encapsulates the process a patient caller undergoes when they dial 111
    and speaks to an operator who triages their call.
    '''
    def __init__(self, identifier, env, args):
        '''
        Constructor method
        
        Params:
        -----\
        identifier: int
            a numeric identifier for the patient.
            
        env: simpy.Environment
            the simulation environment
            
        args: Scenario
            The input data for the scenario
        '''
        self.identifier = identifier
        self.env = env
        
        self.operators = args.operators
        self.call_dist = args.call_dist
        
        self.nurses = args.nurses
        self.nurse_dist = args.nurse_dist
        self.callback_dist = args.callback_dist
        
        self.callback = False
        self.waiting_time_nurse = 0.0
        self.nurse_call_duration = 0.0

    def service(self):
        '''
        Simulates the service process for a call operator
        
        1. request and wait for a call operator
        2. phone triage (triangular)
        3. exit system
        '''
        # record the time that call entered the queue
        start_wait = self.env.now

        # request an operator 
        with self.operators.request() as req:
            yield req
            
            # record the waiting time for call to be answered
            self.waiting_time = self.env.now - start_wait
            trace(f'{self.env.now:.2f}: operator answered call {self.identifier}')
            
            # sample call duration.
            self.call_duration = self.call_dist.sample()
            yield self.env.timeout(self.call_duration)
            
            trace(f'{self.env.now:.2f}: Call {self.identifier} ended ; '
                    + f'waiting time was {self.waiting_time:.2f}')
            
        self.callback = self.callback_dist.sample()
        
        if self.callback:
            
            #record time starting to wait for nurse
            start_wait_nurse = self.env.now
            
            with self.nurses.request() as req:
                yield req
                
                #record the waiting time for nurse
                self.waiting_time_nurse = self.env.now - start_wait_nurse
                trace(f'{self.env.now:.2f}: nurse callback patient {self.identifier}')
                
                self.nurse_call_duration = self.nurse_dist.sample()
                yield self.env.timeout(self.nurse_call_duration)
                
                trace(f'{self.env.now:.2f}: nurse call ended for patient {self.identifier}')

class UrgentCareCallCentre:
    def __init__(self, env, args):
        self.env = env
        self.args = args 
        
        self.patients = []

        args.operators = simpy.Resource(self.env, capacity=args.n_operators)
        args.nurses = simpy.Resource(self.env, capacity=args.n_nurses)
        
    def arrivals_generator(self):
        '''
        IAT is exponentially distributed
        '''
        for caller_count in itertools.count(start=1):
            
            inter_arrival_time = self.args.arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)
            
            trace(f'{self.env.now:.2f} call arrives. Patient ID: {caller_count}')
            
            new_caller = Patient(caller_count, self.env, self.args)
                        
            self.patients.append(new_caller)
            
            self.env.process(new_caller.service())

class Auditor:
    def __init__(self, env, run_length, first_obs=None, interval=None):
        '''
        Auditor Constructor
        
        Params:
        -----\
        env: simpy.Environment
            
        first_obs: float, optional (default=None)
            Time of first scheduled observation.  If none then no scheduled
            audit will take place
        
        interval: float, optional (default=None)
            Time period between scheduled observations. If none then no scheduled
            audit will take place
        '''
        self.env = env
        self.first_observation = first_obs
        self.interval = interval
        self.run_length = run_length
        
        self.queues = []
        self.service = []
        
        # dict to hold states
        self.metrics = {}
        
        # scheduled the periodic audits
        if not first_obs is None:
            env.process(self.scheduled_observation())
            env.process(self.process_end_of_run())
            
    def add_resource_to_audit(self, resource, name, audit_type='qs'):
        if 'q' in audit_type:
            self.queues.append((name, resource))
            self.metrics[f'queue_length_{name}'] = []
        
        if 's' in audit_type:
            self.service.append((name, resource))
            self.metrics[f'system_{name}'] = []           
            
    def scheduled_observation(self):
        '''
        simpy process to control the frequency of 
        auditor observations of the model.  
        
        The first observation takes place at self.first_obs
        and subsequent observations are spaced self.interval
        apart in time.
        '''
        # delay first observation
        yield self.env.timeout(self.first_observation)
        self.record_queue_length()
        self.record_calls_in_progress()
        
        while True:
            yield self.env.timeout(self.interval)
            self.record_queue_length()
            self.record_calls_in_progress()
    
    def record_queue_length(self):
        for name, res in self.queues:
            self.metrics[f'queue_length_{name}'].append(len(res.queue)) 
        
        
    def record_calls_in_progress(self):
        for name, res in self.service:
            self.metrics[f'system_{name}'].append(res.count + len(res.queue)) 
               
        
    def process_end_of_run(self):
        '''
        Create an end of run summary
        
        Returns:
        ---------\
            pd.DataFrame
        '''
        
        yield self.env.timeout(self.run_length - 1)
        
        run_results = {}

        for name, res in self.queues:
            queue_length = np.array(self.metrics[f'queue_length_{name}'])
            run_results[f'mean_queue_{name}'] = queue_length.mean()
            
        for name, res in self.service:
            total_in_system = np.array(self.metrics[f'system_{name}'])
            run_results[f'mean_system_{name}'] = total_in_system.mean()
        
        self.summary_frame = pd.Series(run_results).to_frame()
        self.summary_frame.columns = ['estimate']

# ----------------------------------------------------------------------------
# RESULTS COLLECTION
# ----------------------------------------------------------------------------

def run_results(model, auditor):
    df_results = auditor.summary_frame
    
    # waiting time = sum(waiting times) / no. patients
    mean_waiting_time = np.array([patient.waiting_time 
                                  for patient in model.patients]).mean()

    # operator utilisation = sum(call durations) / (run length X no. operators)
    util = np.array([patient.call_duration 
                     for patient in model.patients]).sum() / \
                    (RUN_LENGTH * N_OPERATORS)
    
    # nurse waiting time 
    nurse_waiting_time = np.array([patient.waiting_time_nurse 
                                  for patient in model.patients
                                  if patient.callback]).mean()   
    
    # nurse utilisation = sum(call durations) / (run length X no. operators)
    nurse_util = np.array([patient.nurse_call_duration 
                     for patient in model.patients if patient.callback]).sum() / \
                    (RUN_LENGTH * N_NURSES)

    # append to results df
    new_row = pd.DataFrame({'estimate':{'mean_wait': mean_waiting_time, 
                                        'ops_util':util,
                                        'mean_nurse_wait':nurse_waiting_time,
                                        'nurse_util':nurse_util}})
    
    # modification for pandas 2.0.0 as Dataframe.append is deprecated.
    df_results = pd.concat([df_results, new_row])
    return df_results
