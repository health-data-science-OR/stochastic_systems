"""
A simple urgent care call centre model built in simpy

This code is used as part of an introduction to
discrete-event similation in Python.

Author: Tom Monks (@TheOpenScienceNerd)

To use the model a script must import `Experiment` and
the functions `single_run()` and/or `multiple_replications()`

Have fun!

"""

import numpy as np
import pandas as pd
import simpy
import itertools

# CONSTANTS AND MODULE LEVEL VARIABLES #####################################

# default resources
N_OPERATORS = 13
N_NURSES = 10

# default mean inter-arrival time (exp)
MEAN_IAT = 60.0 / 100.0

# default service time parameters (triangular)
CALL_LOW = 5.0
CALL_MODE = 7.0
CALL_HIGH = 10.0

# nurse uniform distribution parameters
NURSE_CALL_LOW = 10.0
NURSE_CALL_HIGH = 20.0

# probability of a callback (parameter of Bernoulli)
CHANCE_CALLBACK = 0.4

# sampling settings - we now need 4 streams
N_STREAMS = 4
DEFAULT_RND_SET = 0

# Boolean switch to simulation results as the model runs
TRACE = False

# run variables
WARM_UP_PERIOD = 0
RESULTS_COLLECTION_PERIOD = 1000

# SIMPY SUBCLASSES #########################################


class MonitoredResource(simpy.Resource):
    """
    Subclass of simpy.Resource.

    Based on method described in Law. Simulation Modeling and Analysis 4th Ed.
    Pages 14 - 17

    Calculates both resource utilisation and number in queue.

    """

    def __init__(self, *args, **kwargs):
        # super() is the super class i.e. simpy.Resource
        super().__init__(*args, **kwargs)
        # the time of the last request or release
        self.init_results()

    def init_results(self):

        self.time_last_event = self._env.now

        # the running total of the area under the no. in queue function.
        self.area_n_in_queue = 0.0

        # the running total of the area under the server busy func.
        self.area_resource_busy = 0.0

    def request(self, *args, **kwargs):
        # update time weighted stats BEFORE requesting resource.
        self.update_time_weighted_stats()
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        # update time weighted stats BEFORE releasing resource.
        self.update_time_weighted_stats()
        return super().release(*args, **kwargs)

    def update_time_weighted_stats(self):
        # time since the last release/request
        time_since_last_event = self._env.now - self.time_last_event

        # update last event time
        self.time_last_event = self._env.now

        # update the area under the no. in queue function.
        # len(self.queue) is the number of requests queued.
        self.area_n_in_queue += len(self.queue) * time_since_last_event

        # update the area under the resource busy function.
        # self.count is the number of resources in use.
        self.area_resource_busy += self.count * time_since_last_event

    def end_of_run_cleanup(self, run_length):
        yield self._env.timeout(run_length)

        # update time weighted stats - adds in uncounted resource usage.
        # from last event time until end of simulation run.
        self.update_time_weighted_stats()


# DISTRIBUTION CLASSES ########################################################


class Bernoulli:
    """
    Convenience class for the Bernoulli distribution.
    packages up distribution parameters, seed and random generator.

    The Bernoulli distribution is a special case of the binomial distribution
    where a single trial is conducted

    Use the Bernoulli distribution to sample success or failure.
    """

    def __init__(self, p, random_seed=None):
        """
        Constructor

        Params:
        ------
        p: float
            probability of drawing a 1

        random_seed: int | SeedSequence, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.p = p

    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        float or np.ndarray (if size >=1)
        """
        return self.rand.binomial(n=1, p=self.p, size=size)


class Uniform:
    """
    Convenience class for the Uniform distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, low, high, random_seed=None):
        """
        Constructor

        Params:
        ------
        low: float
            lower range of the uniform

        high: float
            upper range of the uniform

        random_seed: int | SeedSequence, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high

    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        float or np.ndarray (if size >=1)
        """
        return self.rand.uniform(low=self.low, high=self.high, size=size)


class Triangular:
    """
    Convenience class for the triangular distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, low, mode, high, random_seed=None):
        """
        Constructor. Accepts and stores parameters of the triangular dist
        and a random seed.

        Params:
        ------
        low: float
            The smallest values that can be sampled

        mode: float
            The most frequently sample value

        high: float
            The highest value that can be sampled

        random_seed: int | SeedSequence, optional (default=None)
            Used with params to create a series of repeatable samples.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def sample(self, size=None):
        """
        Generate one or more samples from the triangular distribution

        Params:
        --------
        size: int
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        float or np.ndarray (if size >=1)
        """
        return self.rand.triangular(self.low, self.mode, self.high, size=size)


class Exponential:
    """
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, mean, random_seed=None):
        """
        Constructor

        Params:
        ------
        mean: float
            The mean of the exponential distribution

        random_seed: int| SeedSequence, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean

    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        float or np.ndarray (if size >=1)
        """
        return self.rand.exponential(self.mean, size=size)


# EXPERIMENT CLASS ############################################################


class Experiment:
    """
    Encapsulates the concept of an experiment 🧪 with the urgent care
    call centre simulation model.

    An Experiment:
    1. Contains a list of parameters that can be left as defaults or varied
    2. Provides a place for the experimentor to record results of a run
    3. Controls the set & streams of psuedo random numbers used in a run.

    """

    def __init__(
        self,
        random_number_set=DEFAULT_RND_SET,
        n_streams=N_STREAMS,
        n_operators=N_OPERATORS,
        mean_iat=MEAN_IAT,
        call_low=CALL_LOW,
        call_mode=CALL_MODE,
        call_high=CALL_HIGH,
        n_nurses=N_NURSES,
        chance_callback=CHANCE_CALLBACK,
        nurse_call_low=NURSE_CALL_LOW,
        nurse_call_high=NURSE_CALL_HIGH,
    ):
        """
        The init method sets up our defaults.
        """
        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams

        # store parameters for the run of the model
        self.n_operators = n_operators
        self.mean_iat = mean_iat
        self.call_low = call_low
        self.call_mode = call_mode
        self.call_high = call_high

        # resources: we must init resources after an Environment is created.
        # But we will store a placeholder for transparency
        self.operators = None

        # nurse parameters
        self.n_nurses = n_nurses
        self.chance_callback = chance_callback
        self.nurse_call_low = nurse_call_low
        self.nurse_call_high = nurse_call_high

        # nurse resources placeholder
        self.nurses = None

        # initialise results to zero
        self.init_results_variables()

        # initialise sampling objects
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling
        Parameters:
        ----------
        random_number_set: int
            Used to control the set of pseudo random numbers used by
            the distributions in the simulation.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Create the distributions used by the model and initialise
        the random seeds of each.
        """
        # produce n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.random_number_set)
        self.seeds = seed_sequence.spawn(self.n_streams)

        # create distributions

        # call inter-arrival times
        self.arrival_dist = Exponential(
            self.mean_iat, random_seed=self.seeds[0]
        )

        # duration of call triage
        self.call_dist = Triangular(
            self.call_low,
            self.call_mode,
            self.call_high,
            random_seed=self.seeds[1],
        )

        # create the callback and nurse consultation distributions
        self.callback_dist = Bernoulli(
            self.chance_callback, random_seed=self.seeds[2]
        )

        self.nurse_dist = Uniform(
            self.nurse_call_low,
            self.nurse_call_high,
            random_seed=self.seeds[3],
        )

    def init_results_variables(self):
        """
        Initialise all of the experiment variables used in results
        collection.  This method is called at the start of each run
        of the model
        """
        # variable used to store results of experiment
        self.results = {}
        self.results["waiting_times"] = []

        # total operator usage time for utilisation calculation.
        self.results["total_call_duration"] = 0.0

        # nurse sub process results collection
        self.results["nurse_waiting_times"] = []
        self.results["total_nurse_call_duration"] = 0.0

        # reset results collected in montiored resources.
        if self.operators is not None:
            self.operators.init_results()

        if self.nurses is not None:
            self.nurses.init_results()


# SIMPY MODEL LOGIC #########################################################


def trace(msg):
    """
    Turing printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if TRACE:
        print(msg)


def nurse_consultation(identifier, env, args):
    """
    simulates the wait for an consultation with a nurse on the phone.

    1. request and wait for a nurse resource
    2. phone consultation (uniform)
    3. release nurse and exit system

    """
    trace(f"Patient {identifier} waiting for nurse call back")
    start_nurse_wait = env.now

    # request a nurse
    with args.nurses.request() as req:
        yield req

        # record the waiting time for nurse call back
        nurse_waiting_time = env.now - start_nurse_wait
        args.results["nurse_waiting_times"].append(nurse_waiting_time)

        # sample nurse the duration of the nurse consultation
        nurse_call_duration = args.nurse_dist.sample()

        trace(f"nurse called back patient {identifier} at " + f"{env.now:.3f}")

        # schedule process to begin again after call duration
        yield env.timeout(nurse_call_duration)

        trace(
            f"nurse consultation for {identifier}"
            + f" competed at {env.now:.3f}"
        )


def operator_service(identifier, env, args):
    """
    simulates the service process for a call operator

    1. request and wait for a call operator
    2. phone triage (triangular)
    3. release call operator
    4. a proportion of call continue to nurse consultation

    Params:
    ------
    identifier: int
        A unique identifer for this caller

    env: simpy.Environment
        The current environent the simulation is running in
        We use this to pause and restart the process after a delay.

    args: Experiment
        The settings and input parameters for the current experiment

    """

    # record the time that call entered the queue
    start_wait = env.now

    # request an operator - stored in the Experiment
    with args.operators.request() as req:
        yield req

        # record the waiting time for call to be answered
        waiting_time = env.now - start_wait

        # store the results for an experiment
        args.results["waiting_times"].append(waiting_time)
        trace(f"operator answered call {identifier} at " + f"{env.now:.3f}")

        # the sample distribution is defined by the experiment.
        call_duration = args.call_dist.sample()

        # schedule process to begin again after call_duration
        yield env.timeout(call_duration)

        # print out information for patient.
        trace(
            f"call {identifier} ended {env.now:.3f}; "
            + f"waiting time was {waiting_time:.3f}"
        )

    # nurse call back?
    callback_patient = args.callback_dist.sample()

    if callback_patient:
        env.process(nurse_consultation(identifier, env, args))


def arrivals_generator(env, args):
    """
    IAT is exponentially distributed

    Parameters:
    ------
    env: simpy.Environment
        The simpy environment for the simulation

    args: Experiment
        The settings and input parameters for the simulation.
    """
    # use itertools as it provides an infinite loop
    # with a counter variable that we can use for unique Ids
    for caller_count in itertools.count(start=1):

        # rhe sample distribution is defined by the experiment.
        inter_arrival_time = args.arrival_dist.sample()
        yield env.timeout(inter_arrival_time)

        trace(f"call arrives at: {env.now:.3f}")

        # create a operator service process
        env.process(operator_service(caller_count, env, args))


def warmup_complete(warm_up_period, env, args):
    """
    End of warm-up period event. Used to reset results collection variables.

    Parameters:
    ----------
    warm_up_period: float
        Duration of warm-up period in simultion time units

    env: simpy.Environment
        The simpy environment

    args: Experiment
        The simulation experiment that contains the results being collected.
    """
    yield env.timeout(warm_up_period)
    trace(f"{env.now:.2f}: Warm up complete.")

    args.init_results_variables()


#  MODEL WRAPPER FUNCTIONS ##################################################


def single_run(
    experiment,
    rep=0,
    wu_period=WARM_UP_PERIOD,
    rc_period=RESULTS_COLLECTION_PERIOD,
):
    """
    Perform a single run of the model and return the results

    Parameters:
    -----------

    experiment: Experiment
        The experiment/paramaters to use with model

    rep: int
        The replication number.

    wu_period: float, optional (default=WARM_UP_PERIOD)
        The initial transient period of the simulation
        Results from this period are removed from final computations.

    rc_period: float, optional (default=RESULTS_COLLECTION_PERIOD)
        The run length of the model following warm up where results are
        collected.
    """

    # results dictionary.  Each KPI is a new entry.
    run_results = {}

    # reset all results variables to zero and empty
    experiment.init_results_variables()

    # set random number set to the replication no.
    # this controls sampling for the run.
    experiment.set_random_no_set(rep)

    # environment is (re)created inside single run
    env = simpy.Environment()

    # create the MONITORED resources
    experiment.operators = MonitoredResource(env, experiment.n_operators)
    experiment.nurses = MonitoredResource(env, experiment.n_nurses)

    # we pass the experiment to the arrivals generator
    env.process(arrivals_generator(env, experiment))

    # add warm-up period event
    env.process(warmup_complete(wu_period, env, experiment))

    # clean up resources to add in any final resource usage time
    env.process(experiment.operators.end_of_run_cleanup(wu_period + rc_period))
    env.process(experiment.nurses.end_of_run_cleanup(wu_period + rc_period))

    # run for warm-up + results collection period
    env.run(until=wu_period + rc_period)

    # end of run results: calculate mean waiting time
    run_results["01_mean_waiting_time"] = np.mean(
        experiment.results["waiting_times"]
    )

    # end of run results: calculate mean operator utilisation
    # from montiored resource stats
    run_results["02_operator_util"] = (
        experiment.operators.area_resource_busy
        / (rc_period * experiment.n_operators)
    ) * 100.0

    # mean queue lengths
    run_results["03_operator_queue_length"] = (
        experiment.operators.area_n_in_queue / rc_period
    )

    # summary results for nurse process

    # end of run results: nurse waiting time
    run_results["04_mean_nurse_waiting_time"] = np.mean(
        experiment.results["nurse_waiting_times"]
    )

    run_results["05_nurse_util"] = (
        experiment.nurses.area_resource_busy
        / (rc_period * experiment.n_nurses)
    ) * 100.0

    run_results["06_nurse_queue_length"] = (
        experiment.nurses.area_n_in_queue / rc_period
    )

    # return the results from the run of the model
    return run_results


def multiple_replications(
    experiment,
    wu_period=WARM_UP_PERIOD,
    rc_period=RESULTS_COLLECTION_PERIOD,
    n_reps=5,
):
    """
    Perform multiple replications of the model.

    Params:
    ------
    experiment: Experiment
        The experiment/paramaters to use with model

    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.
        the number of minutes to run the model to collect results

    n_reps: int, optional (default=5)
        Number of independent replications to run.

    Returns:
    --------
    pandas.DataFrame
    """

    # loop over single run to generate results dicts in a python list.
    results = [
        single_run(experiment, rep, wu_period, rc_period)
        for rep in range(n_reps)
    ]

    # format and return results in a dataframe
    df_results = pd.DataFrame(results)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = "rep"

    return df_results


def run_all_experiments(
    experiments,
    wu_period=WARM_UP_PERIOD,
    rc_period=RESULTS_COLLECTION_PERIOD,
    n_reps=5,
):
    """
    Run each of the scenarios for a specified results
    collection period and replications.

    Params:
    ------
    experiments: dict
        dictionary of Experiment objects

    rc_period: float
        model run length

    """
    print("Model experiments:")
    print(f"No. experiments to execute = {len(experiments)}\n")

    experiment_results = {}
    for exp_name, experiment in experiments.items():

        print(f"Running {exp_name}", end=" => ")
        results = multiple_replications(
            experiment, wu_period, rc_period, n_reps
        )
        print("done.\n")

        # save the results
        experiment_results[exp_name] = results

    print("All experiments are complete.")

    # format thje results
    return experiment_results

def create_experiments(df_experiments):
    '''
    Returns dictionary of Experiment objects based on contents of a dataframe

    Params:
    ------
    df_experiments: pandas.DataFrame
        Dataframe of experiments. First two columns are id, name followed by 
        variable names.  No fixed width

    Returns:
    --------
    dict
    '''
    experiments = {}
    
    # experiment input parameter dictionary
    exp_dict = df_experiments[df_experiments.columns[1:]].T.to_dict()
    # names of experiments
    exp_names = df_experiments[df_experiments.columns[0]].T.to_list()
    
    print(exp_dict)
    print(exp_names)

    # loop through params and create Experiment objects.
    for name, params in zip(exp_names, exp_dict.values()):
        print(name)
        experiments[name] = Experiment(**params)
    
    return experiments