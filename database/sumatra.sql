/*
 * Load the 2004 Sumatra earthquake data
 */
load data local infile 'sumatra_idcx_arrival.csv' into table idcx_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_leb_origin.csv' into table leb_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_leb_assoc.csv' into table leb_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_sel3_origin.csv' into table sel3_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_sel3_assoc.csv' into table sel3_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

--load data local infile 'sumatra_isc.csv' into table isc_events fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n' ignore 1 lines;

-- Dec 26, 2004 to Dec 29, 2004
insert into dataset values ('sumatra', 1104019200, 1104278400);
commit;

