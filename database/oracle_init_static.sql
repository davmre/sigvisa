create table static_siteid (                                                                                                                                                                                                               
    id          number(*,0),
    sta         varchar2(6),
    lat         number,     
    lon         number,     
    elev        number,   
    staname     varchar2(50), 
    statype     varchar2(2),  
    primary key (id),
    unique (sta));

create table static_phaseid (
  id      number(*,0),
  phase   varchar2(20),
  timedef varchar2(1),
  primary key (id),
  unique  (phase));

create table dataset (
 label       varchar2(20),
 start_time  number,
 end_time    number);

insert into dataset values ("validation", 1237680000, 1238284800);
insert into dataset values ("test", 1238284800, 1238889600);
insert into dataset values ("training", 1238889600, 1245456000);