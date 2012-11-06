import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import Event
from signals.io import load_event_station
from signals.template_models.paired_exp import PairedExpTemplateModel

from database.dataset import *
from database.signal_data import *



class TestRuns(unittest.TestCase):

    def setUp(self):
        self.s = Sigvisa()
        self.run_name = "unit_test_case"

    def test_runid(self):
        s = self.s
        run_name = self.run_name

        iteration, runid = get_latest_fitting_iteration(s.cursor, run_name)
        self.assertIsNone(runid)

        # create a new run (starting at iteration 1)
        iteration, runid = insert_new_fitting_iteration(s.cursor, run_name)
        self.assertEqual(iteration, 1)
        self.assertIsNotNone(iteration, runid)

        # retrieve info on the run and check that it's correct
        (run_name2, iteration2) = get_fitting_run_info(s.cursor, runid)
        self.assertEqual(run_name, run_name2)
        self.assertEqual(iteration, iteration2)

        runid2 = get_fitting_runid(s.cursor, run_name, iteration)
        self.assertEqual(runid, runid2)

        # create a new iteration for the same run, should be the 2nd iteration
        iteration3, runid3 = insert_new_fitting_iteration(s.cursor, run_name)
        self.assertEqual(iteration3, 2)
        self.assertNotEqual(runid3, runid2)

        iteration4, runid4 = get_latest_fitting_iteration(s.cursor, run_name)
        self.assertEqual(runid3, runid4)
        self.assertEqual(iteration3, iteration4)


    def tearDown(self):
        # erase our tracks
        sql_query = "delete from sigvisa_coda_fitting_runs where run_name='%s'" % self.run_name
        self.s.cursor.execute(sql_query)


class TestTemplateParams(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.event = Event(evid=5301405)
        self.tm =  PairedExpTemplateModel(run_name = "", model_type="dummy")

    def test_template_params(self):
        s = Sigvisa()
        phases = s.arriving_phases(self.event, 'URZ')
        param_vals = np.array( \
        [[  1.23891796e+09,   3.92119688e+00,   2.60271377e+00,  -2.80721271e-02], \
         [  1.23891795e+09,   5.21551589e+00,  -1.30253346e-10,  -3.83261311e-02], \
         [  1.23891803e+09,   2.00674254e+00,   1.94897642e-10,  -1.13486495e-04], \
         [  1.23891815e+09,   2.55467272e+00,   1.66756417e-10,  -4.56272392e-02], \
         [  1.23891813e+09,   1.13087039e+02,   3.36043492e+00,  -8.51671938e-02], \
         [  1.23891826e+09,   9.79702075e+00,  -2.85349225e-12,  -8.43354825e-02]])

        (i, runid) = insert_new_fitting_iteration(s.cursor, "unit_test_fake"):

        wave = self.seg['BHZ']
        store_template_params(wave, (phases, param_vals), method_str='unit_test_fake', iid=True, fit_cost=0, run_name="unit_test_fake", iteration=i)

        (phases2, fit_params2), fit_cost = load_template_params(self.event.evid, 'URZ', 'BHZ', 'freq_2.0_3.0', run_name = 'unit_test_fake', iteration=i)
        self.assertEqual(tuple(phases), tuple(phases2))
        self.assertAlmostEqual(np.sum((param_vals-fit_params2).flatten()), 0, places=3)
        self.assertAlmostEqual(fit_cost, 0)

        s.cursor.execute("delete from sigvisa_coda_fits where optim_method='unit_test_fake'")
        s.cursor.execute("delete from sigvisa_coda_fitting_runs where run_name='unit_test_fake'")

#        s.dbconn.commit()

if __name__ == '__main__':
    unittest.main()
