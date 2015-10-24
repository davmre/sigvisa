#!/bin/bash

# to be executed on a cloud host

source ~sigvisa/.bash_profile
source ~sigvisa/.virtualenvs/sigvisa/bin/activate

killall python
rm -r ~sigvisa/sigvisa_log.txt
rm -r ~sigvisa/python/sigvisa/logs/mcmc/*
rm -r ~sigvisa/python/sigvisa/db_cache/*
cd ~sigvisa/python/sigvisa
git fetch
git merge origin/master
