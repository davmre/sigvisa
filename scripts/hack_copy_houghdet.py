from sigvisa import Sigvisa

s = Sigvisa()
new_runid=25

models = s.sql("select sta, phase, phase_context, model_fname from sigvisa_hough_detection_model where fitting_runid=24")
for sta, phase, phase_context, model_fname in models:
    s.sql("insert into sigvisa_hough_detection_model (fitting_runid, sta, phase, phase_context, model_fname) values (%d, '%s', '%s', '%s', '%s')" % (new_runid, sta, phase, phase_context, model_fname))
s.dbconn.commit()
