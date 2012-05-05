load data
 infile 'static_phaseid.csv'
 into table static_phaseid
 fields terminated by "," optionally enclosed by '"'
 ( id, phase, timedef )