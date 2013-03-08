from sigvisa import Sigvisa

from sigvisa.models.wiggles.fourier_features import FourierFeatureNode

def load_wiggle_node_by_family(family_name, len_s, logscale, model_waveform, **kwargs):

    logscale_str = 't' if logscale else 'f'
    srate = model_waveform['srate']
    npts = len_s * srate + 1

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute("select basisid from sigvisa_wiggle_basis where family_name='%s' and npts=%d and srate=%.2f and logscale='%s'" % (family_name, npts, srate, logscale_str))
    basisids = cursor.fetchall()
    cursor.close()

    if len(basisids) == 0:

        if family_name.startswith("fourier_"):

            fundamental = 1.0 / len_s
            min_freq = fundamental
            max_freq = float(family_name.split('_')[1])

            wm_node = FourierFeatureNode(fundamental = fundamental, min_freq=min_freq, max_freq=max_freq, srate = srate, logscale=logscale, npts=npts, model_waveform=model_waveform, family_name=family_name, **kwargs)

        else:
            raise Exception("unsupported wiggle family name %s" % family_name)

    elif len(basisids) == 1:
        basisid = basisids[0][0]
        wm_node = load_wiggle_node(basisid=basisid, model_waveform=model_waveform, **kwargs)
    else:
        raise Exception("something weird is going on with the DB! got multiple basisids %s for fourier family %s" % (basisids, family_name))


    return wm_node

def load_wiggle_node(basisid, **kwargs):
    cursor = Sigvisa().dbconn.cursor()
    sql_query = "select family_name, basis_type, srate, npts, logscale, dimension, fundamental, min_freq, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname from sigvisa_wiggle_basis where basisid=%d" % (basisid,)
    cursor.execute(sql_query)
    basis_info = cursor.fetchone()
    cursor.close()
    family_name, basis_type, srate, npts, logscale, dimension, fundamental, min_freq, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname = basis_info
    logscale = (logscale.lower().startswith('t'))

    if basis_type == "fourier":
        wm_node = FourierFeatureNode(basisid = basisid, fundamental = fundamental, min_freq=min_freq, max_freq=max_freq, srate = srate, logscale=logscale, npts=npts, family_name=family_name, **kwargs)
        assert(wm_node.dimension() == dimension)
    else:
        raise NotImplementedError('wiggle basis %s not yet implemented' % basis_type)

    return wm_node
