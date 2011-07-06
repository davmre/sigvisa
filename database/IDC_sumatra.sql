insert into idcx_arrival select * from arrival_sumatra;
commit;

insert into sel3_origin select * from origin_sel3_sumatra;
commit;

insert into sel3_assoc select * from assoc_sel3_sumatra;
commit;

--insert into leb_origin select * from origin_leb_sumatra;
--commit;

--insert into leb_assoc select * from assoc_leb_sumatra;
--commit;


-- Dec 26, 2004 to Dec 29, 2004
insert into dataset values ('sumatra', 1104019200, 1104278400);
commit;

