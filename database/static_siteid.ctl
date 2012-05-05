load data
 infile 'static_siteid.csv'
 into table static_siteid
 fields terminated by "," optionally enclosed by '"'
 ( id, sta, lat, lon, elev, staname, statype )