import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import get_event
from signals.io import load_event_station
from models.templates.paired_exp import PairedExpTemplateModel

from database.dataset import *
from database.signal_data import *



class TestRuns(unittest.TestCase):

    def setUp(self):
        self.s = Sigvisa()
        self.run_name = "unit_test_case"

    def test_runid(self):
        s = self.s
        cursor = s.dbconn.cursor()
        run_name = self.run_name

        iters = read_fitting_run_iterations(cursor, run_name)
        self.assertTrue(len(iters)==0)

        # create a new run (starting at iteration 1)
        runid = get_fitting_runid(cursor, run_name, 1)
        self.assertEqual(1, get_last_iteration(cursor, run_name))
        # check that the second call returns the same thing (doesn't create a second run)
        runid2 = get_fitting_runid(cursor, run_name, 1)
        self.assertEqual(runid, runid2)

        # retrieve info on the run and check that it's correct
        (run_name2, iteration2) = read_fitting_run(cursor, runid)
        self.assertEqual(run_name, run_name2)
        self.assertEqual(1, iteration2)

        runid3 = get_fitting_runid(cursor, run_name, 2)
        self.assertNotEqual(runid3, runid2)

        iters2 = read_fitting_run_iterations(cursor, run_name)
        self.assertEqual(len(iters2), 2)

    def tearDown(self):
        # erase our tracks
        cursor = self.s.dbconn.cursor()
        sql_query = "delete from sigvisa_coda_fitting_runs where run_name='%s'" % self.run_name
        cursor.execute(sql_query)


class TestTemplateParams(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.event = get_event(evid=5301405)
        self.tm =  PairedExpTemplateModel(run_name = "", model_type="dummy")

    def test_template_params(self):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        phases = s.arriving_phases(self.event, 'URZ')
        param_vals = np.array( \
        [[  1.23891796e+09,   3.92119688e+00,   2.60271377e+00,  -2.80721271e-02], \
         [  1.23891795e+09,   5.21551589e+00,  -1.30253346e-10,  -3.83261311e-02], \
         [  1.23891803e+09,   2.00674254e+00,   1.94897642e-10,  -1.13486495e-04], \
         [  1.23891815e+09,   2.55467272e+00,   1.66756417e-10,  -4.56272392e-02], \
         [  1.23891813e+09,   1.13087039e+02,   3.36043492e+00,  -8.51671938e-02], \
         [  1.23891826e+09,   9.79702075e+00,  -2.85349225e-12,  -8.43354825e-02]])

        run_name = "unit_test_fake"
        i = get_last_iteration(cursor, run_name) + 1
        runid = get_fitting_runid(cursor, run_name, i)

        wave = self.seg['BHZ']
        store_template_params(wave, (phases, param_vals), method_str='unit_test_fake', iid=True, fit_cost=0, run_name="unit_test_fake", iteration=i)

        (phases2, fit_params2), fit_cost, fitid = load_template_params(cursor, self.event.evid, 'URZ', 'BHZ', 'freq_2.0_3.0', run_name = 'unit_test_fake', iteration=i)
        self.assertEqual(tuple(phases), tuple(phases2))
        self.assertAlmostEqual(np.sum((param_vals-fit_params2).flatten()), 0, places=3)
        self.assertAlmostEqual(fit_cost, 0)

        cursor.execute("delete from sigvisa_coda_fits where optim_method='unit_test_fake'")
        cursor.execute("delete from sigvisa_coda_fitting_runs where run_name='unit_test_fake'")

#        s.dbconn.commit()

if __name__ == '__main__':
    unittest.main()
