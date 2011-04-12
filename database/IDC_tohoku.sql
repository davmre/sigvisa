-- adjustments made to include tohoku events as well
drop synonym leb_origin;
create view leb_origin as select * from leb.origin union all select * from
 tohoku_origin_leb;

drop synonym leb_assoc;
create view leb_assoc as select * from leb.assoc union all select * from 
tohoku_assoc_leb;

drop synonym idcx_arrival;
create view idcx_arrival as select * from idcx.arrival union all select *
from arrival_tohoku;

drop synonym sel3_origin;
create view sel3_origin as select * from sel3.origin union all select * from
 tohoku_origin_sel3;

drop synonym sel3_assoc;
create view sel3_assoc as select * from sel3.assoc union all select * from 
tohoku_assoc_sel3;

insert into dataset values ('tohoku', 1299801600, 1299974400);
commit;

