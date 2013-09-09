import numpy as np

from sigvisa.graph.nodes import DeterministicNode
from sigvisa.graph.graph_utils import get_parent_value
import sigvisa.source.brune_source as brune
import sigvisa.source.mm_source as mm

def amp_transfer(ev, band, phase, coda_height):
    return coda_height - ev_source_amp(ev, band, phase)

def ev_source_amp(ev, band, phase):
    if ev.natural_source:
        return brune.source_logamp(mb=ev.mb, band=band, phase=phase)
    else:
        return mm.source_logamp(mb=ev.mb, band=band, phase=phase)


class CodaHeightNode(DeterministicNode):
    # a codaHeightNode is the descendent of an amp_transfer node.
    # it adds in the event source amplitude, deterministically

    def __init__(self, eid, sta, band, chan, phase, initial_value=None, **kwargs):
        self.band=band
        self.chan=chan
        self.phase=phase

        super(CodaHeightNode, self).__init__(**kwargs)

        pv = self._parent_values()
        self.parent_amp_transfer_key, _ = get_parent_value(eid, phase, sta, chan=chan, band=band, param_name="amp_transfer", parent_values = pv, return_key=True)
        self.parent_mb_key = '%d;mb' % eid
        self.parent_naturalsource_key = '%d;natural_source' % eid

        if initial_value is not None:
            self.set_value(initial_value)

    def deriv_value_wrt_parent(self, value=None, key=None, parent_values=None, parent_key=None):
        if parent_values is None:
            parent_values = self._parent_values()
        if value is None:
            value = self.get_value()

        key = key if key else self.single_key
        parent_key = parent_key if parent_key else self.parent_amp_transfer_key

        if key != self.single_key:
            raise AttributeError("don't know how to compute derivative of %s at coda height node" % key)
        if parent_key != self.parent_amp_transfer_key:
            raise AttributeError("don't know how to compute coda height derivative with respect to parent %s" % parent_key)

        return 1.0

    def _ev_source_amp(self, parent_values):
        if parent_values[self.parent_naturalsource_key]:
            return brune.source_logamp(mb=parent_values[self.parent_mb_key], band=self.band, phase=self.phase)
        else:
            return mm.source_logamp(mb=parent_values[self.parent_mb_key], band=self.band, phase=self.phase)

    def compute_value(self, parent_values=None):
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()

        amp_transfer = parent_values[self.parent_amp_transfer_key]
        event_source_amp = self._ev_source_amp(parent_values)

        self._dict[self.single_key] = amp_transfer+event_source_amp
        for child in self.children:
            child.parent_keys_changed.add((self.single_key, self))

    def default_parent_key(self):
        return self.parent_amp_transfer_key

    def invert(self, value, parent_key, parent_values=None):
        if parent_values is None:
            parent_values = self._parent_values()

        if parent_key == self.parent_mb_key:
            raise NotImplementedError("can't invert coda height with respect to event magnitude")
        elif parent_key == self.parent_amp_transfer_key:
            event_source_amp = self._ev_source_amp(parent_values)
            return value - event_source_amp
        else:
            raise KeyError("can't invert coda height with respect to %s" % parent_key)
