from sigvisa.learn.train_coda_models import get_shape_training_data
import numpy as np

X, y, evids = get_shape_training_data(runid=4, site="AS12", chan="SHZ", band="freq_2.0_3.0", phases=["P",], target="amp_transfer", max_acost=np.float("inf"), min_amp=-2)
np.savetxt("X.txt", X)
np.savetxt("y.txt", y)
np.savetxt("evids.txt", evids)
