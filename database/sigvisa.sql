/*drop table sigvisa_gsrun_tmodel;
drop table sigvisa_gsrun_wave;
drop table sigvisa_gridsearch_run;
drop table sigvisa_template_param_model;
drop table sigvisa_wiggle;
drop table sigvisa_coda_fit_phase;
drop table sigvisa_coda_fit;
drop table sigvisa_coda_fitting_run;
*/

create table sigvisa_coda_fitting_run (
 runid 	     int,
 run_name 	varchar(255),
 iter 		int,
 primary key(runid)
);

create table sigvisa_coda_fit (
 fitid    int not null auto_increment, /* MYSQL version */
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
 nmid int not null,
 primary key(fitid),
 foreign key(runid) REFERENCES sigvisa_coda_fitting_run(runid),
 foreign key(nmid) REFERENCES sigvisa_noise_model(nmid))
);


create table sigvisa_coda_fit_phase (
 fpid  int not null auto_increment, /* MYSQL version */
 fitid int not null,
 phase	  varchar(20) not null,
 template_model   varchar(20) default 'paired_exp',
 param1	  double precision,
 param2 double precision,
 param3 double precision,
 param4  double precision,
 amp_transfer double precision,
 wiggle_stime double precision,
 wiggle_fname varchar(255),
 primary key(fpid),
 foreign key (fitid) REFERENCES sigvisa_coda_fit(fitid)
);


create table sigvisa_wiggle (
 wiggleid  int not null auto_increment, /* MYSQL version */
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


create table sigvisa_template_param_model (
 modelid int not null auto_increment,
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


create table sigvisa_gridsearch_run (
 gsid  int not null auto_increment, /* MYSQL version */
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
 gswid int not null auto_increment, /* Oracle version */
 gsid int not null,
 nmid int not null,
 sta varchar(10) not null,
 chan varchar(10) not null,
 band varchar(15) not null,
 hz float(24) not null,
 stime double precision not null,
 etime double precision not null,
 primary key (gswid),
 foreign key (gsid) REFERENCES sigvisa_gridsearch_run(gsid),
 foreign key(nmid) REFERENCES sigvisa_noise_model(nmid))
);

create table sigvisa_gsrun_tmodel (
 gsmid int not null auto_increment,
 gswid int not null,
 modelid int not null,
 primary key (gsmid),
 foreign key (gswid) REFERENCES sigvisa_gsrun_wave(gswid),
 foreign key (modelid) REFERENCES sigvisa_template_param_model(modelid)
);

create table sigvisa_noise_model (
 nmid int not null auto_increment,
 timestamp double precision not null,
 sta varchar(10) not null,
 chan varchar(10) not null,
 band varchar(15) not null,
 hz float(24) not null,
 window_stime double precision not null,
 window_len double precision not null,
 model_type varchar(15) not null,
 nparams int not null,
 mean double precision not null,
 std double precision not null,
 fname varchar(255) not null,
 created_for_hour int not null,
 primary key (nmid)
);
CREATE INDEX noise_hour_idx ON sigvisa_noise_model (created_for_hour);

