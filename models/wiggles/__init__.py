from sigvisa import Sigvisa

from sigvisa.models.wiggles.fourier_features import FourierFeatures


def load_basis_by_family(family_name, srate, runid=None, sta=None, chan=None, band=None, phase=None):

    if family_name.startswith("fourier"):

        s = Sigvisa()

        min_freq = float(band.split('_')[1])
        max_freq = float(band.split('_')[2])

        cursor = s.dbconn.cursor()
        cursor.execute("select basisid from sigvisa_wiggle_basis where family_name='%s' and min_freq=%.2f and max_freq=%.2f and srate=%.2f" % (family_name, min_freq, max_freq, srate))
        basisids = cursor.fetchall()
        cursor.close()

        if len(basisids) == 0:

            fundamental = float(family_name.split('_')[1])
            logscale = "logscale" in family_name

            featurizer = FourierFeatures(fundamental = fundamental, min_freq=min_freq, max_freq=max_freq, srate = srate, logscale=logscale, family_name = family_name)
            basisid = featurizer.save_to_db(s.dbconn)
        elif len(basisids) == 1:
            basisid = basisids[0][0]
            featurizer = load_basis(basisid)
        else:
            raise Exception("something weird is going on with the DB! got multiple basisids %s for fourier family %s" % (basisids, family_name))

    else:
        raise Exception("unsupported wiggle family name %s" % family_name)

    return featurizer, basisid

def load_basis(basisid):
    cursor = Sigvisa().dbconn.cursor()
    sql_query = "select srate, logscale, family_name, dimension, fundamental, min_freq, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname from sigvisa_wiggle_basis where basisid=%d" % (basisid,)
    cursor.execute(sql_query)
    basis_info = cursor.fetchone()
    cursor.close()
    logscale = (basis_info[1].lower().startswith('t'))

    if basis_info[2].startswith("fourier_"):
        from sigvisa.models.wiggles.fourier_features import FourierFeatures
        featurizer = FourierFeatures(fundamental = basis_info[4], min_freq=basis_info[5], max_freq=basis_info[6], srate = basis_info[0], logscale=logscale, family_name=basis_info[2])
    else:
        raise NotImplementedError('wiggle basis %s not yet implemented' % basis_info[2])

    return featurizer
