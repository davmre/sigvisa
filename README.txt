This package contains the VISA implementation. To get started.

- First, load the data in a mysql database into a schema ctbt3mos
  by running the commands in ctbt3mos.sql

- Next, train the priors by calling "python learn.py"

- Now, compile the sources by calling "python setup.py build_ext --inplace"

- Finally, to do inference and store the results into the database
  call "python netvisa.py"

Directory Layout
================

netvisa/

  README.txt
  netvisa.blog

  database/
    ctbt3mos.sql
    visa.sql
      
  priors/

    NumEventPrior.c (.h)
    NumEventPrior.py
    EventLocationPrior.c (.h)
    EventLocationPrior.py

  utils/
    geog.c
    rkvector.c
    interp.c
    counter.py

  visualize/
    earth.py
    makemovie.py

  config/

    NumEventPrior.txt
    EventLocationPrior.txt

    ttime/
      IASPEI.P
      IASPEI.S
      ...

  setup.py

  data.py

  learn.py

  netvisa.py

  netvisa.c

  
  
