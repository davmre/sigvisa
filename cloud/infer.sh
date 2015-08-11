#/bin/bash

source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd $SIGVISA_HOME
python experiments/ctf_real.py $@ >> /home/sigvisa/sigvisa_log.txt 2>&1
