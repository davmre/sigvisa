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
