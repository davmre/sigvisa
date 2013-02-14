

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


create table idcx_amplitude (
 ampid       int not null,
 arid        int not null,
 parid       int,
 chan        varchar(8),
 amp         double,
 per         double,
 snr         double,
 amptime     double,
 start_time  double,
 duration    double,
 bandw       double,
 amptype     varchar(8),
 units       varchar(15),
 clip        varchar(1),
 inarrival   varchar(1),
 auth        varchar(15),
 lddate      datetime,

 primary key (ampid),
 index (arid)
) engine = myisam;

load data local infile 'idcx_amplitude.csv' into table idcx_amplitude fields
terminated by ', ' optionally enclosed by '"' ignore 1 lines;


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

/* NetVISA uses only the arrivals which have a valid azimuth, slowness, snr */
create or replace view idcx_arrival_net as
select * from idcx_arrival idcx where delaz > 0 and delslo > 0 and snr > 0;
