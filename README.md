[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/health-data-science-OR/stochastic_systems/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4333664.svg)](https://doi.org/10.5281/zenodo.4333664)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100+/)

---

# 🏥 Modelling Stochastic Healthcare Systems

Practical materials for discrete event simulation of stochastic health systems, part of the **HPDM097: Making a Difference with Health Data** module.[

> Note: I maintain and update the materials in this repository once a year around December time.

## 📖 Overview

This repository provides hands-on training in discrete event simulation (DES) for healthcare operations research. Using Python and SimPy, you'll learn to build, analyze, and optimize realistic models of complex healthcare systems with stochastic behavior, from emergency departments to surgical scheduling.[1]

### What You'll Learn

- Discrete event simulation fundamentals with SimPy
- Stochastic process modeling for healthcare systems
- Statistical input modeling and distribution fitting
- Time-dependent arrival patterns (non-stationary processes)
- Resource scheduling and capacity planning
- Simulation output analysis and variance reduction
- Real-world simulation modelling and analysis.

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Core Skills**: Basic Python programming, pandas, NumPy
- **Statistics**: Understanding of probability distributions, statistical inference
- **Recommended**: Understanding of Object Orientated Programming

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/health-data-science-OR/stochastic_systems.git
   cd stochastic_systems
   ```

2. **Create the conda environment**
   ```bash
   conda env create -f binder/environment.yml
   conda activate hds_stoch
   ```

3. **Launch Jupyter**
   ```bash
   jupyter lab
   ```

## 📚 Course Structure

### Exercise 1: Introduction to SimPy

**Topics**: SimPy fundamentals, process-based simulation, basic resource modeling

**Materials**:
- Core: Introduction to SimPy notebook
- Optional: Advanced methods for results collection
- Solutions: Complete worked examples

**Key Skills**: 
- Building simple DES models
- Process and resource definition
- Event scheduling and time management
- Basic results collection

**Applications**: Simple call centre models, single-server queues

***

### Exercise 2: Modelling Complex Health Systems

**Topics**: Multi-stage patient pathways, multiple resource types, patient routing

**Materials**:
- Exercise: Complex healthcare system modeling
- Solutions: Full implementation examples

**Key Skills**:
- Multi-resource coordination
- Patient flow modeling
- Complex routing logic
- System-level performance metrics

**Applications**: Multi-stage treatment pathways, minor injury clinic department flow

---

### Exercise 3: Input Modelling

**Topics**: Statistical distribution fitting, goodness-of-fit testing, parameterization from data

#### 3.1 Introduction to auto_fit

**Focus**: Automated distribution fitting tool for healthcare data

**Key Skills**:
- Automated distribution selection
- Parameter estimation
- Visual diagnostic tools

#### 3.2 A&E Data Wrangling and Input Modelling

**Focus**: Real-world emergency department data preparation and analysis

**Key Skills**:
- Healthcare data cleaning and preparation
- Extracting inter-arrival and service time distributions
- Handling missing data and outliers
- Distribution selection for real data

**Applications**: Emergency department arrival patterns, treatment time modeling

***

### Exercise 4: Modelling Time-Dependent Arrivals

**Topics**: Non-stationary Poisson processes, time-varying arrival rates, daily/weekly patterns

**Materials**:
- Exercise: Implementing time-dependent arrival processes
- Solutions: Complete implementation with validation

**Key Skills**:
- Non-homogeneous Poisson processes
- Thinning algorithms
- Modeling daily/weekly seasonality
- Capacity planning for variable demand

**Applications**: ED arrivals by time of day, seasonal demand patterns

***

### Exercise 5: Health Systems with Scheduling Functions

**Topics**: Appointment scheduling, resource allocation, planned vs. unplanned demand

**Materials**:
- Case study: Realistic scheduling scenario
- Solutions: Full implementation and analysis

**Key Skills**:
- Combining scheduled and emergency arrivals
- Appointment booking systems
- Schedule disruption modeling
- Capacity balancing

**Applications**: Outpatient clinics, surgical scheduling, mixed demand systems

***

### Exercise 6: Simulation Output Analysis

**Topics**: Warm-up periods, run length determination, confidence intervals, multiple replications

**Materials**:
- Exercise: Statistical analysis of simulation outputs
- Solutions: Complete analysis workflow

**Key Skills**:
- Identifying and handling transient behavior
- Determining appropriate run lengths
- Calculating valid confidence intervals
- Variance reduction techniques
- Comparing system configurations

***

## 🛠️ Key Technologies

- **SimPy**: Discrete event simulation framework (primary tool)
- **pandas** & **NumPy**: Data manipulation and numerical computing
- **scipy.stats**: Statistical distributions and hypothesis testing
- **matplotlib** Visualization and result presentation

## 💡 Example Applications

Real healthcare scenarios covered in the exercises:

- **Emergency Departments**: Patient flow, triage, treatment pathways
- **Urgent Care Call Centres**: Random arrivals and queueing
- **Mental health services**: Appointment scheduling, no-show modelling


## 📊 Simulation Capabilities

By completing this course, you'll be able to model:

- **Arrivals**: Exponential, time-dependent, scheduled appointments
- **Service Times**: Multiple distributions (exponential, lognormal, empirical)
- **Resources**: Staff, beds, equipment with capacity constraints
- **Patient Routing**: Priority systems, pathways, re-entry
- **Scheduling**: Appointments, shifts, planned maintenance
- **Performance Metrics**: Waiting times, utilization, throughput, queue lengths

## 🎯 Learning Outcomes

After completing these exercises, you will be able to:

1. Build discrete event simulation models of healthcare systems using SimPy
2. Fit statistical distributions to real healthcare data
3. Model time-varying demand patterns
4. Implement scheduling and resource allocation logic
5. Conduct rigorous statistical analysis of simulation outputs
6. Apply simulation for healthcare decision support and optimisation

## 📖 Recommended Reading

- **SimPy Documentation**: [https://simpy.readthedocs.io/](https://simpy.readthedocs.io/)
- **Discrete-Event System Simulation** (Banks et al.)
- **Simulation Modeling and Analysis** (Law & Kelton)
- **Healthcare Operations Management** (Langabeer & Helton)

## 🤝 Contributing

Contributions are welcome! You can help by:

- Reporting bugs or unclear instructions
- Suggesting additional healthcare scenarios
- Adding new exercises or extensions
- Improving documentation and comments
- Sharing your own healthcare simulation models

Please open an issue or submit a pull request.

## 📧 Support & Questions

- **Issues**: Open a GitHub issue for bug reports or technical questions
- **Module**: Part of HPDM097 at [Institution Name]
- **Discussions**: Use GitHub Discussions for general questions

## 📄 Citation

If you use these materials in your research, teaching, or practice, please cite appropriately:

```bibtex
@software{monks_stochastic_systems,
  author = {Monks, Thomas},
  title = {Practical material for modelling stochastic health systems},
  year = 2022,
  publisher = {GitHub},
  url = {https://github.com/health-data-science-OR/stochastic_systems}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Resources

- [health-data-science-OR organization](https://github.com/health-data-science-OR)
- [Forecasting health service demand](https://github.com/health-data-science-OR/forecasting) (companion course)
- [SimPy official documentation](https://simpy.readthedocs.io/)
- [HSMA Programme](https://arc-swp.nihr.ac.uk/training/hsma/) - NHS health service modeling

## 🌟 Acknowledgments

Developed for the Making a Difference with Health Data module. Thanks to all contributors and students who have provided feedback and improvements to these materials.[1]