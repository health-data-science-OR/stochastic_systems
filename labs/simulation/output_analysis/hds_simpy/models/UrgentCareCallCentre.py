'''
A simple urgent care call centre model 
used for learning simpy.

Main model class: UrgentCareCallCentre 
Patient process class: Patient

Process overview:
1. Patient calls 111 and waits (FIFO) for a call centre operator to pickup
2. Patient undergoes call service
3. A percentage of patients require a callback from a nurse.
4. Patients requiring a callback queue FIFO for a nurse callback.
5. Patient undergo nurse service and then leave.;

'''

import numpy as np
import pandas as pd
import itertools
import simpy
import math
from scipy.stats import t

from hds_simpy.models.distributions import (Exponential, Triangular, Uniform,
                                            Bernoulli)

#declare constants for module...

#DEFAULT RESOURCES
N_OPERATORS = 13
N_NURSES = 11

#default parameters for distributions
ARRIVAL_RATE = 100
MEAN_IAT = 60 / ARRIVAL_RATE
CALL_LOW = 5
CALL_HIGH = 10
CALL_MODE = 7
P_CALLBACK = 0.4
NURSE_LOW = 10
NURSE_HIGH = 20

#Should we show a trace of simulated events?
TRACE = False

#default random number SET
DEFAULT_RNG_SET = 1
N_STREAMS = 10

#scheduled audit intervals in minutes.
AUDIT_FIRST_OBS = 10
AUDIT_OBS_INTERVAL = 5

#default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 1440

def trace(msg):
    '''
    Utility function for printing simulation
    set the TRACE constant to FALSE to 
    turn tracing off.
    
    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if TRACE:
        print(msg)


class Scenario(object):
    '''
    Parameter class for 111 simulation model
    '''
    def __init__(self, random_number_set=DEFAULT_RNG_SET):
        '''
        The init method sets up our defaults.

        Parameters:
        -----------
        random_number_set: int, optional (default=DEFAULT_RNG_SET)
            Set to control the initial seeds of each stream of pseudo
            random numbers used in the model.

        '''
        #resource counts
        self.n_operators = N_OPERATORS
        self.n_nurses = N_NURSES

        #sampling
        self.random_number_set = random_number_set
        self.init_sampling()
    
    def set_random_no_set(self, random_number_set):
        '''
        Controls the random sampling 

        Parameters:
        ----------
        random_number_set: int
            Used to control the set of psuedo random numbers
            used by the distributions in the simulation.
        '''
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''
        #create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)


        #create distributions
        self.arrival_dist = Exponential(MEAN_IAT, random_seed=self.seeds[0])
        self.call_dist = Triangular(CALL_LOW, CALL_MODE, CALL_HIGH, 
                                    random_seed=self.seeds[1])
        self.nurse_dist = Uniform(NURSE_LOW, NURSE_HIGH, 
                                  random_seed=self.seeds[2])
        self.callback_dist = Bernoulli(p=P_CALLBACK, random_seed=self.seeds[3])
        
        



class Patient(object):
    '''
    Encapsulates the process a patient caller undergoes when they dial 111
    and speaks to an operator who triages their call.
    '''
    def __init__(self, identifier, env, args):
        '''
        Constructor method
        
        Params:
        -----
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
        
        self.call_duration = 0.0
        self.waiting_time = 0.0

        self.callback = False
        self.waiting_time_nurse = 0.0
        self.nurse_call_duration = 0.0


    def service(self):
        '''
        simualtes the service process for a call operator
        
        1. request and wait for a call operator
        2. phone triage (triangular)
        3. exit system
        '''
        #record the time that call entered the queue
        start_wait = self.env.now

        #request an operator 
        with self.operators.request() as req:
            yield req
            
            #record the waiting time for call to be answered
            self.waiting_time = self.env.now - start_wait
            trace(f'operator answered call {self.identifier} at '
                  + f'{self.env.now:.3f}')
            
            #sample call duration.
            self.call_duration = self.call_dist.sample()
            yield self.env.timeout(self.call_duration)
            
            trace(f'call {self.identifier} ended {self.env.now:.3f}; '
                    + f'waiting time was {self.waiting_time:.3f}')
            
        self.callback = self.callback_dist.sample()
        
        if self.callback:
            
            #record time starting to wait for nurse
            start_wait_nurse = self.env.now
            
            with self.nurses.request() as req:
                yield req
                
                #record the waiting time for nurse
                self.waiting_time_nurse = self.env.now - start_wait_nurse
                trace(f'nurse callback {self.identifier} at '
                    + f'{self.env.now:.3f}')
                
                self.nurse_call_duration = self.nurse_dist.sample()
                yield self.env.timeout(self.nurse_call_duration)
                
                trace(f'nurse call {self.identifier} ended {self.env.now:.3f}')
                

        

class UrgentCareCallCentre(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.init_resources()
        self.patients = []

    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object
        '''
        self.args.operators = simpy.Resource(self.env, 
                                             capacity=self.args.n_operators)
        self.args.nurses = simpy.Resource(self.env, 
                                          capacity=self.args.n_nurses)
    
    def run(self, results_collection_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
            warm_up=0):
        '''
        Conduct a single run of the model in its current 
        configuration

        run length = results_collection_period + warm_up

        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD

        warm_up, float, optional (default=0)
            length of initial transient period to truncate
            from results.

        Returns:
        --------
            None

        '''
        #setup the arrival process
        self.env.process(self.arrivals_generator())
                
        #run
        self.env.run(until=results_collection_period+warm_up)
        
    def arrivals_generator(self):
        '''
        IAT is exponentially distributed
        '''
        for caller_count in itertools.count(start=1):
            
            #iat
            inter_arrival_time = self.args.arrival_dist.sample()
            yield self.env.timeout(inter_arrival_time)
            
            trace(f'call {caller_count} arrives at: {self.env.now:.3f}')
            
            new_caller = Patient(caller_count, self.env, self.args)

            #store the patient
            self.patients.append(new_caller)
            
            #start the patient service process
            self.env.process(new_caller.service())




class Auditor(object):
    def __init__(self, env, run_length=DEFAULT_RESULTS_COLLECTION_PERIOD,
                 first_obs=None, interval=None):
        '''
        Auditor Constructor
        
        Params:
        -----
        env: simpy.Environment
            
        first_obs: float, optional (default=None)
            Time of first scheduled observation.  If none then no scheduled
            audit will take place
        
        interval: float, optional (default=None)
            Time period between scheduled observations. 
            If none then no scheduled audit will take place
        '''
        self.env = env
        self.first_observation = first_obs
        self.interval = interval
        self.run_length = run_length
        
        self.queues = []
        self.service = []
        
        #dict to hold states
        self.metrics = {}
        
        #scheduled the periodic audits
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
        #delay first observation
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
        ---------
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



def run_results(model, auditor):
    df_results = auditor.summary_frame
    
    #waiting time = sum(waiting times) / no. patients
    mean_waiting_time = np.array([patient.waiting_time 
                                  for patient in model.patients]).mean()


    #operator utilisation = sum(call durations) / (run length X no. operators)
    util = np.array([patient.call_duration 
                     for patient in model.patients]).sum() / \
                    (DEFAULT_RESULTS_COLLECTION_PERIOD * N_OPERATORS)
    
    #nurse waiting time 
    nurse_waiting_time = np.array([patient.waiting_time_nurse 
                                  for patient in model.patients
                                  if patient.callback]).mean()
   
    
    #nurse utilisation = sum(call durations) / (run length X no. operators)
    nurse_util = np.array([patient.nurse_call_duration 
                     for patient in model.patients if patient.callback]).sum() / \
                    (DEFAULT_RESULTS_COLLECTION_PERIOD * N_NURSES)

    #append to results df
    new_row = pd.DataFrame({'estimate':{'mean_wait': mean_waiting_time, 
                                        'ops_util':util,
                                        'mean_nurse_wait':nurse_waiting_time,
                                         'nurse_util':nurse_util}})
    df_results = df_results.append(new_row, ignore_index=False)
    return df_results

if __name__ == '__main__':
    # model parameters
    RUN_LENGTH = 1000
    FIRST_OBS = 10
    OBS_INTERVAL = 5
    
    #create simpy environment
    env = simpy.Environment()
    args = Scenario()

    #create model
    model = UrgentCareCallCentre(env, args)

    #create model auditor
    auditor = Auditor(env, RUN_LENGTH, FIRST_OBS, OBS_INTERVAL)
    auditor.add_resource_to_audit(args.operators, 'ops')
    auditor.add_resource_to_audit(args.nurses, 'nurse')

    #run the model
    model.run(RUN_LENGTH)

    print(f'end of run. simulation clock time = {env.now}')
    print('\nSingle run results\n-------------------')
    results_summary = run_results(model, auditor)
    print(results_summary)