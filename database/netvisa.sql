use ctbt3mos;

/* tables to write results */
create table visa_run (
 runid           int auto_increment,

 run_start       datetime,
 run_end         datetime,

 numsamples      int,
 window          int,
 step            int,
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
 timeres  double,
 azres    double,
 slores   double,
 
 index (runid, orid, arid)
) engine = myisam;


/* some utility functions */
delimiter |

create function dist_deg(lon1 double, lat1 double, lon2 double, lat2 double)
returns double deterministic
begin
declare dist double;
set dist = degrees(acos(sin(radians(lat1)) * sin(radians(lat2))
                   + cos(radians(lat1)) * cos(radians(lat2))
                     * cos(radians(lon2 - lon1))));
return dist;
end|

create function dist_km(lon1 double, lat1 double, lon2 double, lat2 double)
returns double deterministic
begin
declare dist double;
set dist = 6371.0 * (acos(sin(radians(lat1)) * sin(radians(lat2))
                     + cos(radians(lat1)) * cos(radians(lat2))
                     * cos(radians(lon2 - lon1))));
return dist;
end|

create function degdiff(from_angle double, to_angle double)
returns double deterministic
begin
  declare angle double;
  set angle = ((to_angle - from_angle) + 360) % 360;
  if angle > 180 then set angle = angle - 360;
  end if;
  return angle;
end|

delimiter ;

create or replace view visa_vs_leb as select run.runid runid, leb.orid
leb, vo.orid visa, round(vo.score) score, round(dist_km(leb.lon,
leb.lat, vo.lon, vo.lat)) err_km, round(leb.lon) lon, round(leb.lat)
lat, round(leb.time) time, round(leb.mb,1) mb from visa_run run join
leb_origin leb on leb.time between run.data_start and run.data_end
left join visa_origin vo on dist_deg(leb.lon, leb.lat, vo.lon, vo.lat)
<= 5 and abs(vo.time-leb.time)<=50 and vo.runid=run.runid;

create or replace view sel3_vs_leb as select run.runid runid, leb.orid
leb, sel3.orid sel3, round(dist_km(leb.lon, leb.lat, sel3.lon,
sel3.lat)) err_km, round(leb.lon) lon, round(leb.lat) lat,
round(leb.time) time, round(leb.mb,1) mb from visa_run run join
leb_origin leb on leb.time between run.data_start and run.data_end
left join sel3_origin sel3 on dist_deg(leb.lon, leb.lat, sel3.lon,
sel3.lat) <= 5 and abs(sel3.time-leb.time)<=50 and sel3.time between
run.data_start and run.data_end;

/* format the output of visa_origin to trim decimal points */
create or replace view visa_origin_deb as select runid, orid, round(lon,1) lon, round(lat,1) lat, round(depth,1) depth, round(time,1) time, round(mb,1) mb, round(score,1) score from visa_origin;

/* join visa_assoc with idcx_arrival */
create or replace view visa_assoc_deb as select runid, orid, phase, sta, round(timeres,1) tres, round(azres,1) azres, round(slores,1) slores, arid, round(time,1) time from visa_assoc join idcx_arrival using (arid);

create or replace view leb_origin_deb as select orid, round(lon,1) lon, round(lat,1) lat, round(depth,1) depth, round(time,1) time, round(mb,1) mb from leb_origin;

/* join leb_assoc with leb_arrival */
create or replace view leb_assoc_deb as select orid, phase, sta, round(timeres,1) tres, round(azres,1) azres, round(slores,1) slores, arid, round(time,1) time from leb_assoc join leb_arrival using (sta, arid);

grant select,insert,update on visa_run to ctbt@localhost;
grant select,insert on visa_origin to ctbt@localhost;
grant select,insert on visa_assoc to ctbt@localhost;
grant select on visa_vs_leb to ctbt@localhost;
grant execute on ctbt3mos.* to ctbt@localhost;

