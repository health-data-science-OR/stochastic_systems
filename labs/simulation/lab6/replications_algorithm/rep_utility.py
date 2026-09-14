"""
module: output_analysis

Provides methods for the confidence interval method for selecting
the number of replications to run with a Discrete-Event Simulation.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import t
import warnings

from typing import Protocol, runtime_checkable, Optional

OBSERVER_INTERFACE_ERROR = "Observers of OnlineStatistics must implement " \
+ "ReplicationObserver interface. i.e. " \
+ "update(results: OnlineStatistics) -> None"


@runtime_checkable
class ReplicationObserver(Protocol):
    """
    Interface for an observer of an instance of the ReplicationsAnalyser
    """
    def update(self, results) -> None:
        """
        Add an observation of a replication

        Parameters:
        -----------
        results: OnlineStatistic
            The current replication to observe.
        """
        pass


class OnlineStatistics:
    """
    Welford’s algorithm for computing a running sample mean and
    variance. Allowing computation of CIs and half width % deviation
    from the mean.

    This is a robust, accurate and old(ish) approach (1960s) that
    I first read about in Donald Knuth’s art of computer programming vol 2.
    """

    def __init__(
        self, 
        data: Optional[np.ndarray] = None, 
        alpha: Optional[float] = 0.1,
        observer: Optional[ReplicationObserver] = None
    ) -> None:
        """
        Initiaise Welford’s algorithm for computing a running sample mean and
        variance. 
        
        Parameters:
        -------
        data: array-like, optional (default = None)
            Contains an initial data sample.

        alpha: float
            To compute 100(1 - alpha) confidence interval

        observer: ReplicationObserver, optional (default=None)
            A user may optionally track the updates to the statistics using a
            ReplicationObserver (e.g. ReplicationTabuliser). This allows further
            tabular or visual analysis or saving results to file if required.

        """
        
        self.n = 0
        self.x_i = None 
        self.mean = None
        # sum of squares of differences from the current mean
        self._sq = None
        self.alpha = alpha
        self._observers = []
        if observer is not None:
            self.register_observer(observer)
        
        if isinstance(data, np.ndarray):
            for x in data:
                self.update(x)

    def register_observer(self, observer: ReplicationObserver) -> None:
        """
        observer: ReplicationRecorder, optional (default = None)
            Include a method for recording the replication results at each 
            update. Part of observer pattern. If None then no results are 
            observed.
            
        """
        if not isinstance(observer, ReplicationObserver):
            raise ValueError(OBSERVER_INTERFACE_ERROR)
        
        self._observers.append(observer)

    @property
    def variance(self) -> float:
        """
        Sample variance of data
        Sum of squares of differences from the current mean divided by n - 1
        """
        
        return self._sq / (self.n - 1)

    @property
    def std(self) -> float:
        """
        Standard deviation of data
        """
        if self.n > 2:
            return np.sqrt(self.variance)
        else:
            return np.nan

    @property
    def std_error(self) -> float:
        """
        Standard error of the mean
        """
        return self.std / np.sqrt(self.n)

    @property
    def half_width(self) -> float:
        """
        Confidence interval half width
        """
        dof = self.n - 1
        t_value = t.ppf(1 - (self.alpha / 2), dof)
        return t_value * self.std_error

    @property
    def lci(self) -> float:
        """
        Lower confidence interval bound
        """
        if self.n > 2:
            return self.mean - self.half_width
        else:
            return np.nan
    
    @property
    def uci(self) -> float:
        """
        Lower confidence interval bound
        """
        if self.n > 2:
            return self.mean + self.half_width
        else:
            return np.nan

    @property
    def deviation(self) -> float:
        """
        Precision of the confidence interval expressed as the
        percentage deviation of the half width from the mean.
        """
        if self.n > 2:
            return self.half_width / self.mean
        else:
            return np.nan

    def update(self, x: float) -> None:
        """
        Running update of mean and variance implemented using Welford's
        algorithm (1962).

        See Knuth. D `The Art of Computer Programming` Vol 2. 2nd ed. Page 216.

        Params:
        ------
        x: float
            A new observation
        """
        self.n += 1
        self.x_i = x
        
        # init values
        if self.n == 1:
            self.mean = x
            self._sq = 0
        else:
            # compute the updated mean
            updated_mean = self.mean + ((x - self.mean) / self.n)

            # update the sum of squares of differences from the current mean
            self._sq += (x - self.mean) * (x - updated_mean)

            # update the tracked mean
            self.mean = updated_mean

        self.notify()

    def notify(self) -> None:
        """
        Notify any observers that a update has taken place.
        """
        for observer in self._observers:
            observer.update(self)


class ReplicationTabulizer:
    """
    Record the replication results from an instance of ReplicationsAlgorithm
    in a pandas DataFrame.
    
    Implement as the part of observer pattern. Provides a summary frame 
    equivalent to the output of a confidence_interval_method
    """
    def __init__(self):
        # to track online stats
        self.stdev = []
        self.lower = []
        self.upper = []
        self.dev = []
        self.cumulative_mean = []
        self.x_i = []
        self.n = 0

    def update(self, results: OnlineStatistics) -> None:
        """
        Add an observation of a replication

        Parameters:
        -----------
        results: OnlineStatistic
            The current replication to observe.
        """
        self.x_i.append(results.x_i)
        self.cumulative_mean.append(results.mean)
        self.stdev.append(results.std)
        self.lower.append(results.lci)
        self.upper.append(results.uci)
        self.dev.append(results.deviation)
        self.n += 1

    def summary_table(self) -> pd.DataFrame:
        """
        Return a dataframe of results equivalent to the confidence interval
        method.
        """
        # combine results into a single dataframe
        results = pd.DataFrame([self.x_i, self.cumulative_mean, 
                                self.stdev, self.lower, self.upper, self.dev]).T
        results.columns = ['Mean', 'Cumulative Mean', 'Standard Deviation', 
                           'Lower Interval', 'Upper Interval', '% deviation']
        results.index = np.arange(1, self.n+1)
        results.index.name = 'replications'

        return results


def confidence_interval_method(
    replications,                          
    alpha: Optional[float] = 0.05, 
    desired_precision: Optional[float] = 0.05, 
    min_rep: Optional[int] = 5, 
    decimal_places: Optional[int] = 2
):
    '''
    The confidence interval method for selecting the number of replications
    to run in a simulation.
    
    Finds the smallest number of replications where the width of the confidence
    interval is less than the desired_precision.  
    
    Returns both the number of replications and the full results dataframe.
    
    Parameters:
    ----------
    replications: arraylike
        Array (e.g. np.ndarray or list) of replications of a performance metric
        
    alpha: float, optional (default=0.05)
        procedure constructs a 100(1-alpha) confidence interval for the 
        cumulative mean.
        
    desired_precision: float, optional (default=0.05)
        Desired mean deviation from confidence interval.
        
    min_rep: int, optional (default=5)
        set to a integer > 0 and ignore all of the replications prior to it 
        when selecting the number of replications to run to achieve the desired
        precision.  Useful when the number of replications returned does not
        provide a stable precision below target.
        
    decimal_places: int, optional (default=2)
        sets the number of decimal places of the returned dataframe containing
        the results
    
    Returns:
    --------
        tuple: int, pd.DataFrame
    
    '''
    # welford's method to track cumulative mean and construct CIs at each rep
    # track the process and construct data table using ReplicationTabuliser
    observer = ReplicationTabulizer()
    stats = OnlineStatistics(alpha=alpha, data=replications[:2], observer=observer)

    # iteratively update.
    for i in range(2, len(replications)):
        stats.update(replications[i])

    results = observer.summary_table()
    
    # get the smallest no. of reps where deviation is less than precision target
    
    try:
        n_reps = results.iloc[min_rep:].loc[results['% deviation'] 
                             <= desired_precision].iloc[0].name
    except IndexError:
        # no replications with desired precision
        message = 'WARNING: the replications do not reach desired precision'
        warnings.warn(message)
        n_reps = -1 
    
    return n_reps, results.round(decimal_places)
        


def plotly_confidence_interval_method(n_reps, conf_ints, metric_name, 
                                   figsize=(1200, 400)):
    """
    Interactive Plotly visualization with deviation hover information
    
    Parameters:
    ----------
    n_reps: int
        Minimum number of reps selected
    conf_ints: pandas.DataFrame
       Results from `confidence_interval_method` function
    metric_name: str
        Name of the performance measure
    figsize: tuple, optional (default=(1200,400))
        Plot dimensions in pixels (width, height)
        
    Returns:
    -------
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Calculate relative deviations [1][4]
    deviation_pct = ((conf_ints['Upper Interval'] - conf_ints['Cumulative Mean']) / 
                     conf_ints['Cumulative Mean'] * 100).round(2)

    # Confidence interval bands with hover info
    for col, color, dash in zip(['Lower Interval', 'Upper Interval'], 
                              ['lightblue', 'lightblue'],
                              ['dot', 'dot']):
        fig.add_trace(go.Scatter(
            x=conf_ints.index,
            y=conf_ints[col],
            line=dict(color=color, dash=dash),
            name=col,
            text=[f'Deviation: {d}%' for d in deviation_pct],
            hoverinfo='x+y+name+text'
        ))

    # Cumulative mean line with enhanced hover
    fig.add_trace(go.Scatter(
        x=conf_ints.index,
        y=conf_ints['Cumulative Mean'],
        line=dict(color='blue', width=2),
        name='Cumulative Mean',
        hoverinfo='x+y+name'
    ))

    # Vertical threshold line
    fig.add_shape(
        type='line',
        x0=n_reps,
        x1=n_reps,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='red', dash='dash')
    )

    # Configure layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        yaxis_title=f'Cumulative Mean: {metric_name}',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig
