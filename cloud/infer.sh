#/bin/bash

export HOME=/home/sigvisa
source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd $SIGVISA_HOME
python $@ 
