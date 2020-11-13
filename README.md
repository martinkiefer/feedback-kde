# Self-Tuning Kernel Density Estimators

This repository contains a modified version of Postgres (9.3.1) that
uses self-optimizing Kernel Density Estimators to compute the 
selectivity of multidimensional range queries on real-valued 
attributes. The estimator relies on query feedback to fine-tune the
model.

Further information about the estimator model (as well as a detailed
evaluation) can be found in our SIGMOD 2015 paper [Self-Tuning, 
GPU-Accelerated Kernel Density Models for Multidimensional Selectivity
Estimation](http://dl.acm.org/citation.cfm?id=2749438).

The estimator uses OpenCL to provide a parallel implementation
that allows accelerated computations on both multi-core CPUs and
graphics cards.
                            
## Prerequisites                                                               
In order to activate this feature, you will need two things:

1. A Linux-based operating system. While the code may compile using OSX as well, the Apple OpenCL SDK is buggy and will lead to incorrect results.

2. An OpenCL-compatible device (e.g. a graphics card or any reasonably modern CPU) and a respective driver SDK. Here are some pointers where you can find one for your device:
    * NVIDIA 

        Install both the latest graphics driver and the CUDA SDK from:

        * www.nvidia.com/Download/index.aspx
        * www.nvidia.com/object/cuda_home_new.html

    * AMD

        Install the latest graphics driver and the APP SDK from:

        * http://support.amd.com/
        * http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/

    * Intel
        
        Install the latest Intel OpenCL SDK from:
                          
        * https://software.intel.com/en-us/articles/opencl-drivers

3. The NLOpt library (http://ab-initio.mit.edu/wiki/index.php/NLopt)
                            
## Configuration & Installation. 
You need to `./configure` Postgres with the new `--with-opencl` flag. This
enables the compilation of all code that depends on OpenCL.

In order to specify the location of your OpenCL SDK, you can use the
`--with-opencl_dir=/PATH/TO/SKD/ROOT` flag.

After configuration has finished, build with `make && make install`.

You can modify whether the estimator uses single- or double-precision
floating point numbers by changing the definiton `kde_float_t` in 
`src/backend/optimizer/path/gpukde/ocl_utilities.h:30`

                      
## Using the estimator 

The estimator is controlled via Postgres configuration variables. You
can set the variables from the SQL prompt via:

    SET <variable> TO <value>;
All variables are valid for the current session only.

###  General parameters
* ocl_use_gpu (boolean, default: true)
> Controls whether the GPU (true) or CPU (false) is used for the
   KDE estimator.
* kde_debug (boolean, default: false)
> If enabled, additional debug information are written to stdout.
* kde_estimation_quality_logfile (string)
>If set, the estimation errors for all KDE estimates are logged to this file.   
* kde_sample_maintenance(default: CAR)
> Specifies the algorithm to maintain the sample under changes.
>> Possible values:
>>	
>> CAR (Correlated Acceptance/Rejection): Correlated Acceptance/Rejection
>>
>> TKR (Triggered Karma Replacement): Resample tuples exceeding a specified Karma threshold
>>
>> PKR (Periodic Karma Replacement): Resample the tuple with the worst Karma periodically
>>
>> PRR (Periodic Random Replacement): Resample a random sample point periodically
>>
>> None: No sample maintenance at all

###  KDE-model specific paramters

* kde_enable (boolean, default: false)
> Controls whether the KDE-estimator is enabled or not.
* kde_samplesize (integer, default: 4300)
> Controls the model size (in rows) that is used for a KDE estimator.

###  Generic optimization parameters
* kde_error_metric (default: RELATIVE)
> Specifies which error metric is optimized by the estimator.
> Possible values are: ABSOLUTE, RELATIVE, QUADRATIC, SQUARED_Q, SQUARED_RELATIVE
* kde_bandwidth_representation (default: Plain)
>   Controls whether the bandwidth is storend and optimized in plain or 
>   logarithmized representation.
>   Possible values are: Plain, Log

### Parameters specific to bandwidth optimization

* kde_collect_feedback (boolean, default: false)
> Controls whether query feedback is collected. All query feedback is written to the system table pg_kdefeedback. By deleting this table, you can erase collected feedback.
* kde_enable_bandwidth_optimization (boolean, default: false)
> Controls whether the bandwidth should be optimized during model construction based on collected queries.
* kde_optimization_feedback_window (integer, default: -1)
> Controls how many of the most recent queries are used for the bandwidth optimization. If set to -1, all queries wil be used.

#### Parameters specific to adaptive bandwidth optimization
* kde_enable_adaptive_bandwidth (boolean, default: false)
> Controls wheter the bandwidth should be optimzed adaptively
> based on incoming queries.
* kde_minibatch_size (integer, default: 5)
> Controls how large (in queries) the mini-batches are that are
> used in the adaptive bandwidth optimization.

### Parameters specific to Karma-based sample maintenance algorithms
* kde_sample_maintenance_karma_limit (float, default: 4.0)
> Controls the upper bound on the Karma a tuple in the sample 
> can aggregate.
* kde_sample_maintenance_karma_threshold (float, default: -2.0)
> Controls the lower bound on the Karma of tuples in the sample
> triggering resampling (TKR only).
    
###  Parameters specific to periodic sample maintenance algorithms
* kde_sample_maintenance_period (integer, default: 1)
> Controls the number of queries considered a period.

### Building the estimator
In order to build a KDE-based estimator, you have to set the
coresponding configuration variables and then issue:

    ANALYZE table(col1, col2, ..., cold);
The estimator will then be automatically applied to all matching
queries.

Dropping an existing estimator can be accomplished by deleting the
corresponding row from the system table `pg_kdemodels`.
                         
## Code location                          
The majority of the code resides in the following two folders:

* `src/backend/kde_feedback` Contains the feedback collection framework.
* `src/backend/optimizer/path/gpukde` Contains the code for the estimator.

We also added the scripts for our experiments in the folder `analysis`.
