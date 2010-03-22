This package contains the VISA implementation. To get started.

- First, load the data in a mysql database:
  copy all the csv files into the database directory
  change to the database directory and then run ctbt3mos.sql from mysql

- Next, compile the sources by calling "python setup.py build_ext --inplace"

- Now, train the priors by calling "python learn.py"

- Finally, to do inference and store the results into the database
  call "python infer.py"

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
    score.{c, h}                    -- Compute the log probability of a world
 
  utils/
    geog.c
    rkvector.c
    interp.c
    counter.py

  visualize/
    earth.py
    makemovie.py

  parameters/

    NumEventPrior.txt             (learnt)
    EventLocationPrior.txt        (learnt)

    ttime/
      iaspei91.P
      iaspei91.S
      ...


  
  
