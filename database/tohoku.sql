/*
 * Load the tohoku specific data into the ctbt3mos database for testing
 */
load data local infile 'tohoku_idcx_arrival.csv' into table idcx_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_leb_origin.csv' into table leb_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_leb_assoc.csv' into table leb_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_sel3_origin.csv' into table sel3_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_sel3_assoc.csv' into table sel3_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_isc.csv' into table isc_events fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n' ignore 1 lines;

insert into dataset values ('tohoku', 1299801600, 1299974400);
commit;
