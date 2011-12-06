This package contains the SIG-VISA implementation. Signal-based Vertically
Integrated Seismological Analysis.

- numpy and scipy are required. To test them, run:
  python -m utils.LogisticModel
  Also, matplotlib and Basemap are required for the visualization, and 
  liblogger (http://liblogger.sourceforge.net/) for logging.  

- Load the data in a mysql database:
  copy all the csv files into the database directory
  change to the database directory and then run ctbt3mos.sql from mysql
  Also, run netvisa.sql to load the tables to store results

- Next, compile the sources by calling "python setup.py build_ext --inplace"

- Now, train the priors by calling "python learn.py" and "python
  learn.py --sigvisa"

- To test your model run "python score.py"

- To do inference and store the results into the database

  call "python infer.py --sigvisa -r 1" (to do inference on 1 hour)

  For longer inference you might want to use the handy utility bgjob
  to run jobs in the background, for example:

  utils/bgjob python infer.py

- To analyze the results of a run:
  python analyze.py   (with an opitional run-identifier e.g. -i 23)

- To get details about a missed event or to debug an event:
    python debug.py <runid> leb <leb-orid>
  To understand a predicted event:
    python debug.py <runid> visa <visa-orid>

- To see the waveforms of an event:
  python2.6 -m utils.waveform leb 5288665
 
  or to see the waveform at a station during a timerange
  python2.6 -m utils.wave_sta AS31 1237683300 1237683500

Directory Layout
================

sigvisa/

  README.txt
  netvisa.blog                      -- describes the model in the BLOG language
  learn.py                          -- learn the parameters
  infer.py                          -- infer the events given the detections
  score.py                          -- score the events and detections
  setup.py                          -- compile C files and the python wrapper
  netvisa.c                         -- python wrapper
  analyze.py                        -- analyze the output of a run

  database/
    ctbt3mos.sql                    -- schema of the data
    netvisa.sql                     -- schema for storing the results
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


  
  
