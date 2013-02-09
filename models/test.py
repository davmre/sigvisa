import numpy as np

class TestSignalLikelihood(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.event = get_event(evid=5301405)
        self.tm =  PairedExpTemplateModel(run_name = "", model_type="dummy")

    def test_generate(self):
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        bhz_23_template =(('P', 'S'), param_vals)
        template = self.tm.generate_template_waveform(template_params=bhz_23_template, model_waveform = self.seg['BHZ'])

        sampled = self.tm.generate_template_waveform(template_params=bhz_23_template, model_waveform = self.seg['BHZ'], sample=True)

        plotting.plot.plot_waveform(template, logscale=True)
        plt.savefig('template.png')

        plotting.plot.plot_waveform(sampled, logscale=True)
        plt.savefig('sampled.png')

    def test_iid_cost(self):
        smoothed = self.seg.with_filter("smooth")
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        ll = self.tm.waveform_log_likelihood_iid(smoothed['BHZ'], (('P', 'S'), param_vals))
        print ll

        missing_bhz = self.seg.with_filter("smooth")['BHZ']
        missing_bhz.data[50:5000] = ma.masked
        ll_missing = self.tm.waveform_log_likelihood_iid(missing_bhz, (('P', 'S'), param_vals))
        print "missing iid ll", ll_missing

        self.assertGreater(ll_missing, ll)

    def test_likelihood(self):
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        ll = self.tm.waveform_log_likelihood(self.seg['BHZ'], (('P', 'S'), param_vals))
        print ll

        missing_bhz = self.seg['BHZ']
        missing_bhz.data[50:5000] = ma.masked
        ll_missing = self.tm.waveform_log_likelihood(missing_bhz, (('P', 'S'), param_vals))
        print "missing ll", ll_missing

        self.assertGreater(ll_missing, ll)
