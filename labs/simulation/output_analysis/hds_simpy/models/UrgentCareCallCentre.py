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
from joblib import Parallel, delayed

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
DEFAULT_RNG_SET = None
N_STREAMS = 10

#scheduled audit intervals in minutes.
AUDIT_FIRST_OBS = 10
AUDIT_OBS_INTERVAL = 5

#default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 1440

#default number of replications
DEFAULT_N_REPS = 5

#warmup auditing
DEFAULT_WARMUP_AUDIT_INTERVAL = 120

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

        #warm-up
        self.warm_up = 0.0

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
            
            self.operator_service_complete()
            
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
                
                self.nurse_service_complete()
    
    def operator_service_complete(self):
        trace(f'call {self.identifier} ended {self.env.now:.3f}; '
               + f'waiting time was {self.waiting_time:.3f}')
    
    def nurse_service_complete(self):
        trace(f'nurse call {self.identifier} ended {self.env.now:.3f}')


class MonitoredPatient(Patient):
    '''
    Monitor a Patient.  Inherits from Patient
    Implemented using the observer design pattern
    
    A MonitoredPatient notifies its observers that a patient
    process has reached an event 
    1. completing call service
    2. completing nurse service
    '''
    def __init__(self, identifier, env, args, model):
        '''
        Constructor
        
        Params:
        -------
        patient: Patient
            patient process to monitor
            
        auditor: Auditor
            auditor
        '''
        super().__init__(identifier, env, args)
        self._observers = [model]
        
    def register_observer(self, observer):
        self._observers.append(observer)
    
    def notify_observers(self, *args, **kwargs):
        for observer in self._observers: 
            observer.process_event(*args, **kwargs)
    
    def operator_service_complete(self):
        #call the patients operator_service_complete method to execute logic
        super().operator_service_complete()
        
        #passes the patient (self) and a message
        self.notify_observers(self, 'operator_service_complete')
            
    def nurse_service_complete(self):
        #call the patients nurse_service_complete method to execute logic
        super().nurse_service_complete()
        
        #passes the patient (self) and a message
        self.notify_observers(self, 'nurse_service_complete')


class UrgentCareCallCentre(object):
    def __init__(self, args):
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()
        self.patients = []
        
        #running performance metrics:
        self.wait_for_operator = 0.0
        self.wait_for_nurse = 0.0
        self.operator_util = 0.0
        self.nurse_util = 0.0
        self.operator_queue = 0.0
        self.nurse_queue = 0.0

        self.nurse_time_used = 0.0
        self.operator_time_used = 0.0

        self.n_calls = 0
        self.n_callbacks = 0

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
            
            #create monitored patient to update KPIs.
            new_caller = MonitoredPatient(caller_count, self.env, self.args,
                                          self)

            #store the patient
            self.patients.append(new_caller)
            
            #start the patient service process
            self.env.process(new_caller.service())

    def process_event(self, *args, **kwargs):
        '''
        Running calculates each time a Patient process ends
        (when a patient departs the simulation model)
        
        Params:
        --------
        *args: list
            variable number of arguments. This is useful in case you need to
            pass different information for different events
        
        *kwargs: dict
            keyword arguments.  Same as args, but you can is a dict so you can
            use keyword to identify arguments.
        
        '''
        patient = args[0]
        msg = args[1]
        
        #only run if warm up complete
        if self.env.now < self.args.warm_up:
            return

        #there are cleaner ways of implementing this, but 
        #for simplicity it is implemented as an if-then statement
        if msg == 'operator_service_complete':
            self.n_calls += 1
            n = self.n_calls
            
            #running calculation for mean operator waiting time
            self.wait_for_operator += \
                (patient.waiting_time - self.wait_for_operator) / n

            #running calc for mean operator utilisation
            self.operator_time_used += patient.call_duration

            #mean operator queue length
            current_q = len(self.args.operators.queue)
            self.operator_queue += (current_q - self.operator_queue) / n
            
        elif msg == 'nurse_service_complete':
            self.n_callbacks += 1
            n = self.n_callbacks
            
            #running calculation for mean nurse waiting time
            self.wait_for_nurse += \
                (patient.waiting_time_nurse - self.wait_for_nurse) / n

            #running calc for mean nurse utilisation
            self.nurse_time_used += patient.nurse_call_duration

            #mean nurse queue length
            current_q = len(self.args.nurses.queue)
            self.nurse_queue += (current_q - self.nurse_queue) / n
            
            
        #print(self.wait_for_operator)

    def run_summary_frame(self):
        #append to results df
        mean_waiting_time = self.wait_for_operator
        nurse_waiting_time = self.wait_for_nurse

        #adjust util calculations for warmup period
        rc_period = self.env.now - self.args.warm_up
        util = self.operator_time_used / (rc_period * self.args.n_operators)
        nurse_util = self.nurse_time_used / (rc_period * self.args.n_nurses)

        df = pd.DataFrame({'1':{'operator_wait': mean_waiting_time, 
                                'operator_queue': self.operator_queue,
                                'ops_util':util,
                                'nurse_wait':nurse_waiting_time,
                                'nurse_util':nurse_util,
                                'nurse_queue': self.nurse_queue}})
        df = df.T
        df.index.name = 'rep'
        return df




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






def single_run(scenario, 
               rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD, 
               warm_up=0, 
               random_no_set=DEFAULT_RNG_SET):
    '''
    Perform a single run of the model and return the results
    
    Parameters:
    -----------
    
    scenario: Scenario object
        The scenario/paramaters to run
        
    rc_period: int
        The length of the simulation run that collects results
        
    warm_up: int, optional (default=0)
        warm-up period in the model.  The model will not collect any results
        before the warm-up period is reached.  
        
    random_no_set: int or None, optional (default=1)
        Controls the set of random seeds used by the stochastic parts of the 
        model.  Set to different ints to get different results.  Set to None
        for a random set of seeds.
        
    Returns:
    --------
        pandas.DataFrame:
        results from single run.
    '''  
        
    #set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    #create an instance of the model
    model = UrgentCareCallCentre(scenario)

    model.run(results_collection_period=rc_period, warm_up=warm_up)
    
    #run the model
    results_summary = model.run_summary_frame()
    
    return results_summary


def multiple_replications(scenario, 
                          rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
                          warm_up=0,
                          n_reps=DEFAULT_N_REPS, 
                          n_jobs=-1):
    '''
    Perform multiple replications of the model.
    
    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model
    
    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model beyond warm up
        to collect results
    
    warm_up: float, optional (default=0)
        initial transient period.  no results are collected in this period

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.
        
    Returns:
    --------
    List
    '''    
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(scenario, 
                                                      rc_period, 
                                                      warm_up,
                                                      random_no_set=rep) 
                                  for rep in range(n_reps))


    #format and return results in a dataframe
    df_results = pd.concat(res)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results


class WarmupAuditor():
    '''
    Warmup Auditor for the model.
    
    Stores the cumulative means for:
    1. operator waiting time
    2. nurse waiting time
    3. operator utilisation
    4. nurse utilitsation.
    
    '''
    def __init__(self, model, interval=DEFAULT_WARMUP_AUDIT_INTERVAL):
        self.env = model.env
        self.model = model
        self.interval = interval
        self.wait_for_operator = []
        self.wait_for_nurse = []
        self.operator_util = []
        self.nurse_util = []
        
    def run(self, rc_period):
        '''
        Run the audited model
        
        Parameters:
        ----------
        rc_period: float
            Results collection period.  Typically this should be many times
            longer than the expected results collection period.
            
        Returns:
        -------
        None.
        '''
        #set up data collection for warmup variables.
        self.env.process(self.audit_model())
        self.model.run(rc_period, 0)
        
    def audit_model(self):
        '''
        Audit the model at the specified intervals
        '''
        for i in itertools.count():
            yield self.env.timeout(self.interval)

            #Performance metrics
            #calculate the utilisation metrics
            util = self.model.operator_time_used / \
                (self.env.now * self.model.args.n_operators)
            nurse_util = self.model.nurse_time_used / \
                (self.env.now * self.model.args.n_nurses)
            
            #store the metrics
            self.wait_for_operator.append(self.model.wait_for_operator)
            self.wait_for_nurse.append(self.model.wait_for_nurse)
            self.operator_util.append(util)
            self.nurse_util.append(nurse_util)
            
    def summary_frame(self):
        '''
        Return the audit observations in a summary dataframe
        
        Returns:
        -------
        pd.DataFrame
        '''
        
        df = pd.DataFrame([self.wait_for_operator,
                           self.wait_for_nurse,
                           self.operator_util,
                           self.nurse_util]).T
        df.columns = ['operator_wait', 'nurse_wait', 'operator_util',
                      'nurse_util']
        
        return df


def warmup_single_run(scenario, rc_period, 
                      interval=DEFAULT_WARMUP_AUDIT_INTERVAL, 
                      random_no_set=DEFAULT_RNG_SET):
    '''
    Perform a single run of the model as part of the warm-up
    analysis.
    
    Parameters:
    -----------
    
    scenario: Scenario object
        The scenario/paramaters to run
        
    results_collection_period: int
        The length of the simulation run that collects results
               
    audit_interval: int, optional (default=60)
        during between audits as the model runs.
        
    Returns:
    --------
        Tuple:
        (mean_time_in_system, mean_time_to_nurse, mean_time_to_triage,
         four_hours)
    '''        
    #set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    #create an instance of the model
    model = UrgentCareCallCentre(scenario)

    #create warm-up model auditor and run
    audit_model = WarmupAuditor(model, interval)
    audit_model.run(rc_period)

    return audit_model.summary_frame()


#example solution
def warmup_analysis(scenario, rc_period, n_reps=DEFAULT_N_REPS,
                    interval=DEFAULT_WARMUP_AUDIT_INTERVAL,
                    n_jobs=-1):
    '''
    Conduct a warm-up analysis of key performance measures in the model.
    
    The analysis runs multiple replications of the model.
    In each replication a WarmupAuditor periodically takes observations
    of the following metrics:

    metrics included:
    1. Operator waiting time
    2. Nurse callback waiting time
    3. Operator utilisation
    4. Nurse utilisation

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model
    
    rc_period: int
        number of minutes to run the model in simulated time
        
    n_reps: int, optional (default=5)
        Number of independent replications to run.

    n_jobs: int, optional (default=-1)
        Number of processors for parallel running of replications

    Returns:
    --------
    dict of pd.DataFrames where each dataframe related to a metric.
    Each column of a dataframe represents a replication and each row 
    represents an observation.
    '''    
    res = Parallel(n_jobs=n_jobs)(delayed(warmup_single_run)(scenario, 
                                                             rc_period,
                                                             random_no_set=rep,
                                                             interval=interval) 
                                  for rep in range(n_reps))
    
    #format and return results
    metrics = {'operator_wait':[],
           'nurse_wait':[],
           'operator_util':[],
           'nurse_util':[]}

    #preprocess results of each replication
    for rep in res:
        metrics['operator_wait'].append(rep.operator_wait)
        metrics['nurse_wait'].append(rep.nurse_wait)
        metrics['operator_util'].append(rep.operator_util)
        metrics['nurse_util'].append(rep.nurse_util)
        
    #cast to dataframe
    metrics['operator_wait'] = pd.DataFrame(metrics['operator_wait']).T
    metrics['nurse_wait'] = pd.DataFrame(metrics['nurse_wait']).T
    metrics['operator_util'] = pd.DataFrame(metrics['operator_util']).T
    metrics['nurse_util'] = pd.DataFrame(metrics['nurse_util']).T
    
    #index as obs number
    metrics['operator_wait'].index = np.arange(1, 
                                               len(metrics['operator_wait'])+1)
    metrics['nurse_wait'].index = np.arange(1, len(metrics['nurse_wait'])+1)
    metrics['operator_util'].index = np.arange(1, 
                                               len(metrics['operator_util'])+1)
    metrics['nurse_util'].index = np.arange(1, len(metrics['nurse_util'])+1)

    #obs label
    metrics['operator_wait'].index.name = "audit"
    metrics['nurse_wait'].index.name = "audit"
    metrics['operator_util'].index.name = "audit"
    metrics['nurse_util'].index.name = "audit"
    
    #columns as rep number
    cols = [f'rep_{i}' for i in range(1, n_reps+1)]
    metrics['operator_wait'].columns = cols
    metrics['nurse_wait'].columns = cols
    metrics['operator_util'].columns = cols
    metrics['nurse_util'].columns = cols
    
    return metrics
        