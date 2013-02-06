-- Create the VISA tables in the IDC database
-- Connect as the VISA user before running this script
create table dataset (
  label       varchar(20) not null,
  start_time  number not null,
  end_time    number not null
);

/* 01-APR-08 00:00:00 --  01-APR-09 00:00:00 (1 year) */
insert into dataset values ('training', 1207008000, 1238544000);
/* 01-APR-09 00:00:00 --  01-MAY-09 00:00:00 (1 month) */
insert into dataset values ('validation', 1238544000, 1241136000);
/* 01-MAY-09 00:00:00 -- 01-JULY-09 00:00:00 (2 month) */
insert into dataset values ('test', 1241136000, 1246406400);

create synonym leb_arrival for leb_ref.arrival;

create table leb_assoc as select * from leb_ref.assoc;

create table leb_origin as select * from leb_ref.origin;

create table idcx_arrival as select * from idcx_ref.arrival;

create table sel3_assoc as select * from sel3_ref.assoc;

create table sel3_origin as select * from sel3_ref.origin;

create synonym static_site for static_ref.site;

create table static_siteid (
  id          int primary key,
  sta         varchar(6) unique,
  lat         float(24),
  lon         float(24),
  elev        float(24),
  staname     varchar(50),
  statype     char(2)
);

/* only use sites which have at least a 100 P phase detections */
insert into static_siteid select rownum, freq.sta,site.lat, site.lon,site.elev,site.staname,site.statype from (select sta, count(*) p_cnt from leb_assoc where phase='P' and timedef='d' group by sta order by p_cnt desc) freq, static_site site where freq.sta=site.sta and site.offdate=-1 and freq.p_cnt > 100;

commit;

create table static_phaseid (
  id      int primary key,
  phase   varchar(20) unique,
  timedef varchar(1)
);

insert into static_phaseid values (1,'P','d');
insert into static_phaseid values (2,'Pn','d');
insert into static_phaseid values (3,'PKP','d');
insert into static_phaseid values (4,'Sn','d');
insert into static_phaseid values (5,'S','d');
insert into static_phaseid values (6,'PKPbc','d');
insert into static_phaseid values (7,'PcP','d');
insert into static_phaseid values (8,'pP','d');
insert into static_phaseid values (9,'Lg','d');
insert into static_phaseid values (10,'PKPab','d');
insert into static_phaseid values (11,'ScP','d');
insert into static_phaseid values (12,'PKKPbc','d');
insert into static_phaseid values (13,'Pg','d');
insert into static_phaseid values (14,'Rg','d');
insert into static_phaseid values (15,'tx','n');
insert into static_phaseid values (16,'Sx','n');
insert into static_phaseid values (17,'Px','n');
insert into static_phaseid values (18,'N','n');

commit;


/* NetVISA uses only the arrivals which have a valid azimuth, slowness, snr */
create or replace view idcx_arrival_net as
select * from idcx_arrival idcx where delaz > 0 and delslo > 0 and snr > 0;

/* load the ISC events for validation */
create table isc_events (
 eventid     int not null,
 region      varchar(100) not null,
 author      varchar(10) not null,
 lon         float not null,
 lat         float not null,
 depth       float not null,
 time        float not null,
 mb          float not null,
 ndef        int,
 nsta        int,
 gap         int,
 ml          float not null,
 primary key (eventid, author)
);
create index isc_events_time on isc_events(time);
create index isc_events_author_time on isc_events(author, time);

-- sqlldr userid=user/pass control=isc_events.ctl log=isc_events.log skip=1

/* tables to write results */
create table visa_run (
 runid           int,
 run_start       date,
 run_end         date,
 numsamples      int,
 window          int,
 step            int,
 seed            int,
 data_start      float,
 data_end        float,
 score           float(24),
 f1              float(24),
 prec            float(24),
 recall          float(24),
 error_avg       float(24),
 error_sd        float(24),
 descrip         varchar(100),
 primary key(runid)
);

create sequence visa_runid start with 1 increment by 1 nomaxvalue;

create trigger visa_runid_trigger
before insert on visa_run
for each row
begin
select visa_runid.nextval into :new.runid from dual;
end;
/


create table visa_origin (
 runid   int,
 orid    int,
 lon     float(24),
 lat     float(24),
 depth   float(24),
 time    float,
 mb      float(24),
 score   float(24),
 primary key(runid, orid)
);

create table visa_assoc (
 runid    int,
 orid     int,
 phase    varchar(20),
 arid     int,
 score    float(24),
 timeres  float(24),
 azres    float(24),
 slores   float(24),
 primary key(runid, orid, arid)
);

create table sigvisa_coda_fitting_run (
 runid 	     int,
 run_name 	varchar(255),
 iter 		int,
 primary key(runid)
);

create table sigvisa_coda_fit (
 /* fitid    int not null auto_increment, */ /* MYSQL version */
 fitid	  int not null, /* Oracle version */
 runid    int not null,
 evid     int not null,
 sta      varchar(10) not null,
 chan     varchar(10) not null,
 band     varchar(15) not null,
 hz	  float(24),
 optim_method  varchar(1024),
 iid        int,
 stime      double,
 etime      double,
 acost      float(24),
 dist       float(24),
 azi        float(24),
 timestamp float(24),
 elapsed   float(24),
 human_approved varchar(1) default 0,
 primary key(fitid),
 foreign key(runid) REFERENCES sigvisa_coda_fitting_run(runid)
);

/* hack to implement auto_increment in Oracle */
create sequence fitid_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger fitid_trigger
before insert on sigvisa_coda_fit
for each row
begin
select fitid_seq.nextval into :new.fitid from dual;
end;
/

create table sigvisa_coda_fit_phase (
/* fpid  int not null auto_increment, */ /* MYSQL version */
 fpid  int not null, /* Oracle version */
 fitid int not null,
 phase	  varchar(20) not null,
 template_model   varchar(20) default 'paired_exp',
 param1	  double precision,
 param2 double precision,
 param3 double precision,
 param4  double precision,
 amp_transfer double precision,
 wiggle_fname varchar(255),
 primary key(fpid),
 foreign key (fitid) REFERENCES sigvisa_coda_fit(fitid)
);

/* hack to implement auto_increment in Oracle */
create sequence fpid_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger fpid_trigger
before insert on sigvisa_coda_fit_phase
for each row
begin
select fpid_seq.nextval into :new.fpid from dual;
end;
/

create table sigvisa_wiggle (
/* wiggleid  int not null auto_increment, */ /* MYSQL version */
 wiggleid  int not null, /* Oracle version */
 fpid int not null,
 stime double precision not null,
 etime double precision not null,
 srate double precision not null,
 timestamp double precision not null,
 type varchar(31) not null,
 log varchar(1) not null,
 meta0 double precision,
 meta1 double precision,
 meta2 double precision,
 meta3 double precision,
 meta_str varchar(255),
 params blob not null,
 primary key(wiggleid),
 foreign key (fpid) REFERENCES sigvisa_coda_fit_phase(fpid)
);

/* hack to implement auto_increment in Oracle */
create sequence wiggleid_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger wiggleid_trigger
before insert on sigvisa_wiggle
for each row
begin
select wiggleid_seq.nextval into :new.wiggleid from dual;
end;
/

create table sigvisa_template_param_model (
 modelid int not null, /* Oracle version */
 fitting_runid int not null,
 template_shape varchar(15) not null,
 param varchar(15) not null,
 site varchar(10) not null,
 chan varchar(10) not null,
 band varchar(15) not null,
 phase varchar(10) not null,
 max_acost double precision not null,
 min_amp double precision not null,
 require_human_approved varchar(1) not null,
 model_type varchar(31) not null,
 model_fname varchar(255) not null,
 training_set_fname varchar(255) not null,
 n_evids int not null,
 training_ll double precision not null,
 timestamp double precision not null,
 primary key (modelid),
 foreign key (fitting_runid) REFERENCES sigvisa_coda_fitting_run(runid)
);

/* hack to implement auto_increment in Oracle */
create sequence modelid_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger modelid_trigger
before insert on sigvisa_template_param_model
for each row
begin
select modelid_seq.nextval into :new.modelid from dual;
end;
/

create table sigvisa_gridsearch_run (
/* gsid  int not null auto_increment, */ /* MYSQL version */
 gsid  int not null, /* Oracle version */
 evid int not null,
 timestamp double precision not null,
 elapsed double precision not null,
 lon_nw float(24) not null,
 lat_nw float(24) not null,
 lon_se float(24) not null,
 lat_se float(24) not null,
 pts_per_side int not null,
 max_evtime_proposals int not null,
 true_depth varchar(1) not null,
 phases varchar(127) not null,
 likelihood_method varchar(63) not null,
 wiggle_model_type varchar(31) not null,
 heatmap_fname varchar(255) not null,
 primary key (gsid)
);

create table sigvisa_gsrun_wave (
 gswid int not null, /* Oracle version */
 gsid int not null,
 sta varchar(10) not null,
 chan varchar(10) not null,
 band varchar(15) not null,
 hz float(24) not null,
 stime double precision not null,
 etime double precision not null,
 primary key (gswid),
 foreign key (gsid) REFERENCES sigvisa_gridsearch_run(gsid)
);

create table sigvisa_gsrun_tmodel (
 gsmid int not null, /* Oracle version */
 gswid int not null,
 modelid int not null,
 primary key (gsmid),
 foreign key (gswid) REFERENCES sigvisa_gsrun_wave(gswid),
 foreign key (modelid) REFERENCES sigvisa_template_param_model(modelid)
);

/* hack to implement auto_increment in Oracle */
create sequence gsid_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger gsid_trigger
before insert on sigvisa_gridsearch_run
for each row
begin
select gsid_seq.nextval into :new.gsid from dual;
end;
/

/* hack to implement auto_increment in Oracle */
create sequence gsw_id_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger gsw_id_trigger
before insert on sigvisa_gsrun_wave
for each row
begin
select gsw_id_seq.nextval into :new.gswid from dual;
end;
/

/* hack to implement auto_increment in Oracle */
create sequence gsm_id_seq start with 1 increment by 1 nomaxvalue;
create or replace trigger gsm_id_trigger
before insert on sigvisa_gsrun_tmodel
for each row
begin
select gsm_id_seq.nextval into :new.gsmid from dual;
end;
/



/* ESSENTIAL FUNCTIONS */
CREATE OR REPLACE FUNCTION now RETURN DATE
/* now is needed for compatibility with MySQL */
IS
BEGIN
  RETURN sysdate;
END;
/


/* HELPER FUNCTIONS */
CREATE OR REPLACE
        FUNCTION unixts_to_date(unixts IN PLS_INTEGER) RETURN DATE IS
                /**
                 * Converts a UNIX timestamp into an Oracle DATE
                 */
                unix_epoch DATE := TO_DATE('19700101000000','YYYYMMDDHH24MISS');
                max_ts PLS_INTEGER := 2145916799; -- 2938-12-31 23:59:59
                min_ts PLS_INTEGER := -2114380800; -- 1903-01-01 00:00:00
                oracle_date DATE;
                BEGIN
                        IF unixts> max_ts THEN
                                RAISE_APPLICATION_ERROR(
                                        -20901,
                                        'UNIX timestamp too large for 32 bit limit'
                                );
                        ELSIF unixts <min_ts THEN
                                RAISE_APPLICATION_ERROR(
                                        -20901,
                                        'UNIX timestamp too small for 32 bit limit' );
                        ELSE
                                oracle_date := unix_epoch + NUMTODSINTERVAL(unixts, 'SECOND');
                        END IF;
                        RETURN (oracle_date);
END;
/

CREATE OR REPLACE FUNCTION unixts_to_char(unixts IN PLS_INTEGER) RETURN CHAR
IS
/**
 * Converts a UNIX timestamp into a string
 */
BEGIN
  RETURN to_char(unixts_to_date(unixts), 'DD-MON-YY HH24:MI:SS');
END;
/

CREATE OR REPLACE FUNCTION char_to_unixts(datechar IN CHAR) RETURN PLS_INTEGER
IS
/**
 * Converts a UNIX timestamp into a string
 */
BEGIN
  RETURN (TO_DATE(datechar, 'DD-MON-YY') - to_date('01-JAN-1970','DD-MON-YYYY')) * (86400);
END;
/


create or replace procedure dump_table_to_csv( p_tname in varchar2,
                                               p_dir   in varchar2,
                                               p_filename in varchar2 )
is
    l_output        utl_file.file_type;
    l_theCursor     integer default dbms_sql.open_cursor;
    l_columnValue   varchar2(4000);
    l_status        integer;
    l_query         varchar2(1000)
                    default 'select * from ' || p_tname;
    l_colCnt        number := 0;
    l_separator     varchar2(1);
    l_descTbl       dbms_sql.desc_tab;
begin
    l_output := utl_file.fopen( p_dir, p_filename, 'w' );
    execute immediate 'alter session set nls_date_format=''yyyy-mm-dd hh24:mi:ss''';

    dbms_sql.parse(  l_theCursor,  l_query, dbms_sql.native );
    dbms_sql.describe_columns( l_theCursor, l_colCnt, l_descTbl );

    for i in 1 .. l_colCnt loop
        utl_file.put( l_output, l_separator || '"' || l_descTbl(i).col_name || '"' );

        dbms_sql.define_column( l_theCursor, i, l_columnValue, 4000 );
        l_separator := ',';
    end loop;
    utl_file.new_line( l_output );

    l_status := dbms_sql.execute(l_theCursor);

    while ( dbms_sql.fetch_rows(l_theCursor) > 0 ) loop
        l_separator := '';
        for i in 1 .. l_colCnt loop
            dbms_sql.column_value( l_theCursor, i, l_columnValue );
            utl_file.put( l_output, l_separator || l_columnValue );
            l_separator := ',';
        end loop;
        utl_file.new_line( l_output );
    end loop;
    dbms_sql.close_cursor(l_theCursor);
    utl_file.fclose( l_output );

    execute immediate 'alter session set nls_date_format=''dd-MON-yy'' ';
exception
    when others then
        execute immediate 'alter session set nls_date_format=''dd-MON-yy'' ';
        raise;
end;
/
