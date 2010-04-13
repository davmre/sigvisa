use ctbt3mos;

/* tables to write results */
create table visa_run (
 runid           int auto_increment,

 run_start       datetime,
 run_end         datetime,

 numsamples      int,
 seed            int,
 data_start      double,
 data_end        double,

 score           double,

 f1              double,
 prec            double,
 recall          double,
 error_avg       double,
 error_sd        double,

 descrip         varchar(100),

 primary key(runid)
) engine = myisam;

create table visa_origin (
 runid   int,
 orid    int,

 lon     double,
 lat     double,
 depth   double,
 time    double,
 mb      double,

 score   double,

 index (runid, orid)
) engine = myisam;

create table visa_assoc (
 runid    int,
 orid     int,
 phase    varchar(20),
 arid     int,
 score    double,
 index (runid, orid, arid)
) engine = myisam;

grant select,insert,update on visa_run to ctbt@localhost;
grant select,insert on visa_origin to ctbt@localhost;
grant select,insert on visa_assoc to ctbt@localhost;

