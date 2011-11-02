import database.db
from database.dataset import *
import utils.geog

cursor = database.db.connect().cursor()
sites = read_sites(cursor)
ssites = utils.geog.stations_by_distance(-10,-10,sites)

for site in ssites:
    print site
