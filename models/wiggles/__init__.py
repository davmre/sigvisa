from sigvisa import Sigvisa

from sigvisa.models.wiggles.fourier_features import FourierFeatureGenerator


def load_wiggle_generator_by_family(family_name, len_s, srate, **kwargs):

    npts = len_s * srate
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute("select basisid from sigvisa_wiggle_basis where family_name='%s' and npts=%d and srate=%.2f " % (family_name, npts, srate))
    basisids = cursor.fetchall()
    cursor.close()

    if len(basisids) == 0:

        if family_name.startswith("fourier_"):
            max_freq = float(family_name.split('_')[1])
            logscale = "logscale" in family_name
            wg = FourierFeatureGenerator(max_freq=max_freq, srate = srate, logscale=logscale, npts=npts, family_name=family_name, **kwargs)

        else:
            raise Exception("unsupported wiggle family name %s" % family_name)

    elif len(basisids) == 1:
        basisid = basisids[0][0]
        wg = load_wiggle_generator(basisid=basisid, **kwargs)
    else:
        raise Exception("something weird is going on with the DB! got multiple basisids %s for fourier family %s" % (basisids, family_name))

    return wg

def load_wiggle_generator(basisid, **kwargs):
    cursor = Sigvisa().dbconn.cursor()
    sql_query = "select family_name, basis_type, srate, npts, logscale, dimension, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname from sigvisa_wiggle_basis where basisid=%d" % (basisid,)
    cursor.execute(sql_query)
    basis_info = cursor.fetchone()
    cursor.close()
    family_name, basis_type, srate, npts, logscale, dimension, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname = basis_info
    logscale = (logscale.lower().startswith('t'))

    if basis_type == "fourier":
        wg = FourierFeatureGenerator(basisid = basisid, max_freq=max_freq, srate = srate, logscale=logscale, npts=npts, family_name=family_name, **kwargs)
        assert(wg.dimension() == dimension)
    else:
        raise NotImplementedError('wiggle basis %s not yet implemented' % basis_type)

    return wg
