/*
 * Load the 2004 Sumatra earthquake data
 */
load data local infile 'sumatra_idcx_arrival.csv' into table idcx_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_leb_arrival.csv' into table leb_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_leb_origin.csv' into table leb_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_leb_assoc.csv' into table leb_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_sel3_origin.csv' into table sel3_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_sel3_assoc.csv' into table sel3_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'sumatra_wfdisc.csv' into table idcx_wfdisc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

-- the following command works on windows
-- we are removing the first 13 characters '/archive/ops/' changing / to \\
-- and prepending C:\Users\Nimar\Data\var\ctbt_data
-- update idcx_wfdisc set dir = concat('C:\\Users\\Nimar\\Data\\var\\ctbt_data\\', replace(substr(dir,14), '/','\\')) where substr(dir,1,13) = '/archive/ops/';

--load data local infile 'sumatra_isc.csv' into table isc_events fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n' ignore 1 lines;

-- Sunday Dec 26, 2004 00:00 GMT to Wednesday Dec 29, 2004 00:00 GMT
-- day 361 to day 364
insert into dataset values ('sumatra', 1104019200, 1104278400);
commit;

