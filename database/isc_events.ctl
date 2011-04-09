load data infile '200903.csv' infile '200904.csv' infile '200905.csv' truncate
into table isc_events fields terminated by ',' 
(eventid, region enclosed by '"', 
author, lon, lat, depth, time, mb, ndef, nsta, 
gap terminated by whitespace)

