This package contains the NET-VISA implementation. Network Vertically
Integrated Seismological Analysis.

- rpy2, R, and numpy are required. To test rpy2 and R
  run: python -m utils.LogisticModel
  Also, matplotlib and Basemap are required for the visualization

- Load the data in a mysql database:
  copy all the csv files into the database directory
  change to the database directory and then run ctbt3mos.sql from mysql

- Next, compile the sources by calling "python setup.py build_ext --inplace"

- Now, train the priors by calling "python learn.py"

- To test your model run "python score.py"

- Finally, to do inference and store the results into the database
  call "python infer.py" (Not yet implemented)

Directory Layout
================

netvisa/

  README.txt
  netvisa.blog                      -- describes the model in the BLOG language
  learn.py                          -- learn the parameters
  infer.py                          -- infer the events given the detections
  score.py                          -- score the events and detections
  setup.py                          -- compile C files and the python wrapper
  netvisa.c                         -- python wrapper

  database/
    ctbt3mos.sql                    -- schema of the data
    visa.sql                        -- schema for storing the results
    dataset.py                      -- load the dataset

  priors/

    NumEventPrior.{py, c, h}        -- Number of events
    EventLocationPrior.{py, c, h}   -- Location of events
    EventDetectionPrior.{py, c, h}  -- Detection of a phase at a site
    NumFalseDetPrior.{py, c, h}     -- Number of false detections at each site
    ArrivalTimePrior.{py, c, h}     -- Arrival Time at a site
    ArrivalAzimuthPrior.{py, c, h}  -- Arrival Azimuth at a site
    ArrivalSlownessPrior.{py, c, h} -- Arrival Slowness at a site
    ArrivalPhasePrior.{py, c, h}    -- Arrival Phase at a site
    EarthModel.{c, h}               -- Travel time and distances for all phases
    score.{c, h}                    -- Compute the log probability of a world

  utils/
    LogisticModel.py
    geog.py                         -- simple geographical distances etc.

  infer/
    infer.h
    infer.c

  results/
    mwmatching.py                  -- max weighted matching
    compare.py                     -- compare the answer to the (assumed) truth

  visualize/
    earth.py
    makemovie.py

  parameters/

    NumEventPrior.txt             (learnt)
    EventLocationPrior.txt        (learnt)
    EventDetectionPrior.txt       (learnt)
    NumFalseDetPrior.txt          (learnt)
    ArrivelTimePrior.txt          (learnt)
    ArrivelAzimuthPrior.txt       (learnt)
    ArrivelSlownessPrior.txt      (learnt)
    
    ttime/
      iasp91.P
      iasp91.S
      ...


  
  
