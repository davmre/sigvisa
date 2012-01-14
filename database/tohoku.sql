/*
 * Load the tohoku specific data into the ctbt3mos database for testing
 */
load data local infile 'tohoku_idcx_arrival.csv' into table idcx_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_leb_arrival.csv' into table leb_arrival fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_leb_origin.csv' into table leb_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_leb_assoc.csv' into table leb_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_sel3_origin.csv' into table sel3_origin fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_sel3_assoc.csv' into table sel3_assoc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_isc.csv' into table isc_events fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n' ignore 1 lines;

load data local infile 'tohoku_wfdisc.csv' into table idcx_wfdisc fields terminated by ',' lines terminated by '\r\n' ignore 1 lines;

-- the following command works on windows
-- we are removing the first 13 characters '/archive/ops/' changing / to \\
-- and prepending C:\Users\Nimar\Data\var\ctbt_data
-- update idcx_wfdisc set dir = concat('C:\\Users\\Nimar\\Data\\var\\ctbt_data\\', replace(substr(dir,14), '/','\\')) where substr(dir,1,13) = '/archive/ops/';


-- Fri, 11 Mar 2011 00:00:00 GMT to Sun, 13 Mar 2011 00:00:00 GMT
-- day 70 to day 72
insert into dataset values ('tohoku', 1299801600, 1299974400);
commit;
