/* Run this file from a directory which contains the CTBT data in .csv format 
 *
 * Note: the waveform data is assumed to be located in /var/ctbt_data/seismic,
 * /var/ctbt_data/hydro/ etc.
 */

create database ctbt3mos;

use ctbt3mos;

/* dataset table has labels "training", "validation" and "test" */

create table dataset (
 label       varchar(20) not null,
 start_time  double not null,
 end_time    double not null
) engine = myisam;

/* 3/22/09 00:00:00 -- 3/29/09 00:00:00 (7 days) */
insert into dataset values ("validation", 1237680000, 1238284800);
/* 3/29/09 00:00:00 -- 4/5/09 00:00:00 (7 days) */
insert into dataset values ("test", 1238284800, 1238889600);
/* 4/5/09 00:00:00  -- 6/20/09 00:00:00 (76 days) */
insert into dataset values ("training", 1238889600, 1245456000);

/* Japanese events */
create table jap_hdf (
 isol        varchar(3),
 iyr         int,
 mon         int,
 iday        int,
 ihr         int,
 min         int,
 sec         double,
 glat        double,
 glon        double,
 depth       double,
 iscdepth    double,
 mb          double,
 ms          double,
 mw          double,
 ntot        int,
 ntel        int,
 ndep        int,
 igreg       int
) engine = myisam;

load data local infile 'jap2.5.hdf.csv' into table jap_hdf 
fields terminated by ',';

/* add a UTC timestamp to the table */
alter table jap_hdf add column (time double);

update jap_hdf set time = (unix_timestamp(concat(2000+iyr,'-',mon,'-',iday,' ',
ihr,':',min,':',sec)) - unix_timestamp(utc_timestamp()) +
unix_timestamp() + sec - truncate(sec,0));

alter table jap_hdf add index(time);

/* Honshu events */
create table hon_hdf (
 isol        varchar(3),
 iyr         int,
 mon         int,
 iday        int,
 ihr         int,
 min         int,
 sec         double,
 glat        double,
 glon        double,
 depth       double,
 iscdepth    double,
 mb          double,
 ms          double,
 mw          double,
 ntot        int,
 ntel        int,
 ndep        int,
 igreg       int
) engine = myisam;

load data local infile 'hon2.5.hdf.csv' into table hon_hdf 
fields terminated by ',';

/* add a UTC timestamp to the table */
alter table hon_hdf add column (time double);

update hon_hdf set time = (unix_timestamp(concat(2000+iyr,'-',mon,'-',iday,' ',
ihr,':',min,':',sec)) - unix_timestamp(utc_timestamp()) +
unix_timestamp() + sec - truncate(sec,0));

alter table hon_hdf add index(time);

/* table for ISC data */
create table isc_events (
 eventid     int not null,
 region      varchar(100) not null,
 author      varchar(10) not null,
 lon         double not null,
 lat         double not null,
 depth       double not null,
 time        double not null,
 mb          double not null,
 ndef        int,
 nsta        int,
 gap         int,

 primary key (eventid, author),
 index (time),
 index (author, time)
) engine = myisam;

load data local infile '200903.csv' into table isc_events fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

load data local infile '200904.csv' into table isc_events fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

load data local infile '200905.csv' into table isc_events fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

load data local infile '200906.csv' into table isc_events fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

/* leb_arrival */

create table leb_arrival (
 sta         varchar(6) not null,
 time        double not null,
 arid        int,
 jdate       int,
 stassid     int,
 chanid      int,
 chan        varchar(8),
 iphase      varchar(8),
 stype       varchar(1),
 deltim      double,
 azimuth     double,
 delaz       double,
 slow        double,
 delslo      double,
 ema         double,
 rect        double,
 amp         double,
 per         double,
 logat       double,
 clip        varchar(1),
 fm          varchar(2),
 snr         double,
 qual        varchar(1),
 auth        varchar(15),
 commid      int,
 lddate      datetime,
 
 primary key (arid),
 index (time)
) engine = myisam;

load data local infile 'leb_arrival.csv' into table leb_arrival fields 
terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* reb_arrival */

create table reb_arrival (
 sta         varchar(6) not null,
 time        double not null,
 arid        int,
 jdate       int,
 stassid     int,
 chanid      int,
 chan        varchar(8),
 iphase      varchar(8),
 stype       varchar(1),
 deltim      double,
 azimuth     double,
 delaz       double,
 slow        double,
 delslo      double,
 ema         double,
 rect        double,
 amp         double,
 per         double,
 logat       double,
 clip        varchar(1),
 fm          varchar(2),
 snr         double,
 qual        varchar(1),
 auth        varchar(15),
 commid      int,
 lddate      datetime,
 
 primary key (arid),
 index (time)
) engine = myisam;

load data local infile 'reb_arrival.csv' into table reb_arrival fields 
terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* idcx_arrival */

create table idcx_arrival (
 sta         varchar(6) not null,
 time        double not null,
 arid        int not null,
 jdate       int,
 stassid     int,
 chanid      int,
 chan        varchar(8),
 iphase      varchar(8),
 stype       varchar(1),
 deltim      double,
 azimuth     double,
 delaz       double,
 slow        double,
 delslo      double,
 ema         double,
 rect        double,
 amp         double,
 per         double,
 logat       double,
 clip        varchar(1),
 fm          varchar(2),
 snr         double,
 qual        varchar(1),
 auth        varchar(15),
 commid      int,
 lddate      datetime,
 
 primary key (time, arid),
 index (arid)
) engine = myisam;

load data local infile 'idcx_arrival.csv' into table idcx_arrival fields
terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* leb_assoc */

create table leb_assoc (
 arid        int,
 orid        int,
 sta         varchar(6),
 phase       varchar(8),
 belief      double,
 delta       double,
 seaz        double,
 esaz        double,
 timeres     double,
 timedef     varchar(1),
 azres       double,
 azdef       varchar(1),
 slores      double,
 slodef      varchar(1),
 emares      double,
 wgt         double,
 vmodel      varchar(15),
 commid      int,
 lddate      datetime,

 primary key (arid, orid),
 index (orid)
) engine = myisam ;

load data local infile 'leb_assoc.csv' into table leb_assoc fields terminated
by ', ' optionally enclosed by '"' ignore 1 lines;

/* reb_assoc */

create table reb_assoc (
 arid        int,
 orid        int,
 sta         varchar(6),
 phase       varchar(8),
 belief      double,
 delta       double,
 seaz        double,
 esaz        double,
 timeres     double,
 timedef     varchar(1),
 azres       double,
 azdef       varchar(1),
 slores      double,
 slodef      varchar(1),
 emares      double,
 wgt         double,
 vmodel      varchar(15),
 commid      int,
 lddate      datetime,

 primary key (arid, orid),
 index (orid)
) engine = myisam ;

load data local infile 'reb_assoc.csv' into table reb_assoc fields terminated
by ', ' optionally enclosed by '"' ignore 1 lines;

/* sel3_assoc */

create table sel3_assoc (
 arid        int,
 orid        int,
 sta         varchar(6),
 phase       varchar(8),
 belief      double,
 delta       double,
 seaz        double,
 esaz        double,
 timeres     double,
 timedef     varchar(1),
 azres       double,
 azdef       varchar(1),
 slores      double,
 slodef      varchar(1),
 emares      double,
 wgt         double,
 vmodel      varchar(15),
 commid      int,
 lddate      datetime,

 primary key (arid, orid),
 index (orid)
) engine = myisam ;

load data local infile 'sel3_assoc.csv' into table sel3_assoc fields terminated
by ', ' optionally enclosed by '"' ignore 1 lines;
                                    
/* sel3_origin */

create table sel3_origin (
 lat         double not null,
 lon         double not null,
 depth       double not null,
 time        double not null,
 orid        int,
 evid        int,
 jdate       int,
 nass        int,
 ndef        int,
 ndp         int,
 grn         int,
 srn         int,
 etype       varchar(7),
 depdp       double,
 dtype       varchar(1),
 mb          double,
 mbid        int,
 ms          double,
 msid        int,
 ml          double,
 mlid        int,
 algorithm   varchar(15),
 auth        varchar(15),
 commid      int,
 lddate      datetime,

 primary key (orid),
 index (time)
) engine = myisam;

load data local infile 'sel3_origin.csv' into table sel3_origin 
fields terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* leb_origin */

create table leb_origin (
 lat         double not null,
 lon         double not null,
 depth       double not null,
 time        double not null,
 orid        int,
 evid        int,
 jdate       int,
 nass        int,
 ndef        int,
 ndp         int,
 grn         int,
 srn         int,
 etype       varchar(7),
 depdp       double,
 dtype       varchar(1),
 mb          double,
 mbid        int,
 ms          double,
 msid        int,
 ml          double,
 mlid        int,
 algorithm   varchar(15),
 auth        varchar(15),
 commid      int,
 lddate      datetime,

 primary key (orid),
 index (time)
) engine = myisam;

load data local infile 'leb_origin.csv' into table leb_origin fields terminated
by ', ' optionally enclosed by '"' ignore 1 lines;

/* reb_origin */

create table reb_origin (
 lat         double not null,
 lon         double not null,
 depth       double not null,
 time        double not null,
 orid        int,
 evid        int,
 jdate       int,
 nass        int,
 ndef        int,
 ndp         int,
 grn         int,
 srn         int,
 etype       varchar(7),
 depdp       double,
 dtype       varchar(1),
 mb          double,
 mbid        int,
 ms          double,
 msid        int,
 ml          double,
 mlid        int,
 algorithm   varchar(15),
 auth        varchar(15),
 commid      int,
 lddate      datetime,

 primary key (orid),
 index (time)
) engine = myisam;

load data local infile 'reb_origin.csv' into table reb_origin fields terminated
by ', ' optionally enclosed by '"' ignore 1 lines;


/* static_site */

create table static_site (
 sta         varchar(6) not null,
 ondate      int not null,
 offdate     int,
 lat         double,
 lon         double,
 elev        double,
 staname     varchar(50),
 statype     varchar(4),
 refsta      varchar(6),
 dnorth      double,
 deast       double,
 lddate      datetime,

 primary key (sta, ondate)
) engine = myisam;

load data local infile 'static_site.csv' into table static_site fields 
terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* list of sites which detect at least one arrival with az,slo, and snr
associated with a P arrival for an leb event */

/* static_sitechan */

create table static_sitechan (
 sta         varchar(6),
 chan        varchar(8),
 ondate      int,
 chanid      int,
 offdate     int,
 ctype       varchar(4),
 edepth      double,
 hang        double,
 vang        double,
 descrip     varchar(50),
 lddate      datetime,

 primary key (sta, chan, ondate)
) engine = myisam;

load data local infile 'static_sitechan.csv' into table static_sitechan fields 
terminated by ', ' optionally enclosed by '"' ignore 1 lines;

/* wfdisc */

create table idcx_wfdisc (
 sta         varchar(6) not null,
 chan        varchar(8) not null,
 time        double not null,
 wfid        int not null,
 chanid      int,
 jdate       int,
 endtime     double,
 nsamp       int,
 samprate    double,
 calib       double,
 calper      double,
 instype     varchar(6),
 segtype     varchar(1),
 datatype    varchar(2),
 clip        varchar(1),
 dir         varchar(64),
 dfile       varchar(32),
 foff        int,
 commid      int,
 lddate      datetime,

 primary key (wfid),

 index (sta, chan, time, endtime)

) engine = myisam;

load data local infile 'idcx_wfdisc.csv' into table idcx_wfdisc 
fields terminated by ', ' optionally enclosed by '"' ignore 1 lines;
                                    
/* change the location of the data from /archive/ops/ to /var/ctbt_data/ */
update idcx_wfdisc set dir = concat('/var/ctbt_data/', substr(dir,14));

create table static_siteid (
  id          int,
  sta         varchar(6),
  lat         double,
  lon         double,
  elev        double,
  staname     varchar(50),
  statype     char(2),
  primary key (id),
  unique  key(sta))
engine = myisam;

load data local infile 'static_siteid.csv' into table static_siteid fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

create table static_phaseid (
  id      int,
  phase   varchar(20),
  timedef varchar(1),

  primary key (id),
  unique  key (phase))
engine = myisam;

load data local infile 'static_phaseid.csv' into table static_phaseid fields 
terminated by ',' optionally enclosed by '"' ignore 1 lines;

/* NetVISA uses only the arrivals which have a valid azimuth, slowness, snr */
create or replace view idcx_arrival_net as 
select * from idcx_arrival idcx where delaz > 0 and delslo > 0 and snr > 0;

/* create a user for querying the data and give him privileges */
create user ctbt@localhost;
grant select on ctbt3mos.* to ctbt@localhost;
