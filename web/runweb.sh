#!/bin/bash

export HOME=/home/sigvisa
source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd /home/sigvisa/python/sigvisa/web
python manage.py runserver 0.0.0.0:8001
