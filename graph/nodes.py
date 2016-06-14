import numpy as np

import copy
from sigvisa.learn.train_param_common import load_modelid
from sigvisa.models import DummyModel


class Node(object):

    def __init__(self, model=None, label="", initial_value = None, fixed=False, keys=None, children=(), parents = (), low_bound=None, high_bound=None, hack_param_constraint=False, hack_coarse_tts=None, hack_ttr_max=25.0, hack_ampt_z_max=4.0):

        self.hack_param_constraint = hack_param_constraint
        self.hack_coarse_tts = hack_coarse_tts
        self.hack_ttr_max = hack_ttr_max
        self.hack_ampt_z_max = hack_ampt_z_max

        self.model = model
        self._fixed = fixed
        self.label = label
        self.mark = 0
        self.key_prefix = ""

        # built up in sigvisa_graph.setup_site_param_node_indep
        self.params_modeled_jointly = set()

        if not keys:
            if isinstance(initial_value, dict):
                keys = initial_value.keys()
            elif isinstance(fixed, dict):
                keys = fixed.keys()
            else:
                keys = (label,)
        self._keys = sorted(keys)

        if isinstance(fixed, bool):
            self._mutable = { k : not fixed for k in keys }
        elif isinstance(fixed, dict):
            self._mutable = { k : not fixed[k] for k in keys }
        elif fixed is None:
            self._mutable = { k : True for k in keys }
        else:
            raise ValueError("passed invalid fixed-values setting %s" % fixed)
        self._fixed = not any(self._mutable.itervalues())
        self._update_mutable_cache()

        if isinstance(initial_value, float) and not np.isfinite(initial_value):
            raise ValueError("creating node %s with NaN value!" % label)

        if len(keys) > 1:
            if initial_value:
                assert (set(keys) == set(initial_value.iterkeys()) )
            if fixed and isinstance(fixed, dict):
                assert (set(keys) == set(fixed.iterkeys()) )
            self.single_key = None
            self._dict = initial_value
        else:
            self.single_key = keys[0]
            self._dict = dict()
            self._dict[self.single_key] = initial_value

        self._low_bounds = dict()
        self._high_bounds = dict()
        for key in keys:
            self._low_bounds[key] = np.float("-inf")
            self._high_bounds[key] = np.float("inf")

        # TODO: support specifying bounds for multiple-key nodes
        if low_bound is not None:
            self._low_bounds[self.single_key] = low_bound
        if high_bound is not None:
            self._high_bounds[self.single_key] = high_bound

        self.children = set()
        self.child_jgps = []
        self.parents = dict()
        self.parent_keys_changed = set()
        self.parent_nodes_added = set()
        self.parent_keys_removed = set()
        self._pv_cache = dict()

        self.child_set_changed=True
        for child in children:
            self.addChild(child)
        for parent in parents:
            self.addParent(parent)

    def addChild(self, child):
        self.children.add(child)
        for key in self.keys():
            child.parents[key] = self
            child.parent_keys_removed.discard(key)
        self.child_set_changed=True
        child.parent_nodes_added.add(self)


    def addParent(self, parent, stealth=False):
        for key in parent.keys():
            self.parents[key] = parent
            if not stealth:
                self.parent_keys_removed.discard(key)
        if not stealth:
            parent.child_set_changed=True
            self.parent_nodes_added.add(parent)
            parent.children.add(self)

    # NOTE: removeChild and removeParent assume that node is actually
    # being removed from the graph. We'd have to do more bookkeeping
    # to just remove the edge.
    def removeChild(self, child):
        self.children.remove(child)
        self.child_set_changed = True

    def removeParent(self, parent):
        self.parent_nodes_added.discard(parent)

        for key in parent.keys():
            del self.parents[key]
            self.parent_keys_removed.add(key)
            self.parent_keys_changed.discard((key, parent))

    def deterministic(self):
        # deterministic nodes are treated specially (see
        # DeterministicNode below).
        return False

    def _calc_deterministic_children(self):
        # return all nodes that compute a deterministic function of
        # this node, in topologically sorted order.

        # TODO: currently assumes a tree structure to the
        # deterministic children, i.e. doesn't do a true topo
        # sort. Results will be INCORRECT if this is not the case.

        def traverse_child(n):
            if n.deterministic():
                self._deterministic_children.append(n)
                for c in n.children:
                    traverse_child(c)
            else:
                return

        self._deterministic_children = []
        for c in self.children:
            traverse_child(c)

    def _calc_stochastic_children(self):
        # return all stochastic nodes that depend directly on this
        # node or a deterministic function of this node. For each
        # such stochastic child, also include the chain of
        # deterministic nodes connecting it to the given node.

        def traverse_child(n, intermediates):
            if not n.deterministic():
                self._stochastic_children.append((n, intermediates))
                return
            else:
                for c in n.children:
                    traverse_child(c, intermediates + (n,))

        self._stochastic_children = []
        for c in self.children:
            traverse_child(c, ())

    def get_deterministic_children(self):
        if self.child_set_changed:
            self._calc_stochastic_children()
            self._calc_deterministic_children()
            self.child_set_changed=False
        return self._deterministic_children

    def get_stochastic_children(self):
        if self.child_set_changed:
            self._calc_stochastic_children()
            self._calc_deterministic_children()
            self.child_set_changed=False
        return self._stochastic_children

    def keys(self):
        # return the list of keys provided by this node
        return self._keys

    def get_value(self, key=None):
        key = key if key else self.single_key
        return self._dict[key]

    def set_value(self, value, key=None, force_deterministic_consistency=True):

        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("trying to set NaN at node %s" % self.label)

        if isinstance(value, np.ndarray) and value.size == 1:
            raise ValueError("%s: setting scalar value as %s np array %s " % (self.label, value.shape, value))


        #if "tt_residual" in self.label and value != 0:
        #    import pdb; pdb.set_trace()


        key = key if key else self.single_key
        if self._mutable[key]:
            self._dict[key] = value

            for child in self.children:
                child.parent_keys_changed.add((key, self))
            for child in self.child_jgps:
                child.parent_keys_changed.add((key, self))

            if force_deterministic_consistency:
                for child in self.get_deterministic_children():
                    child.parent_predict()
        else:
            raise Exception('trying to set fixed value %s at node %s' % (key, self.label))

    def get_local_value(self, key):
        return self.get_value(self.key_prefix + key)

    def set_local_value(self, value, key, **kwargs):
        return self.set_value(key=self.key_prefix + key, value=value, **kwargs)

    def get_dict(self):
        return self._dict

    def set_dict(self, value, override_fixed=False):
        assert(set(value.iterkeys()) == set(self._mutable.iterkeys()))
        if override_fixed:
            self._dict = value
        else:
            self._dict = {k : value[k] if self._mutable[k] else self._dict[k] for k in value.iterkeys() }


        for child in self.children:
            child.parent_keys_changed.update((k, self) for k in self._mutable_keys)

    def _set_values_from_model(self, value):
        # here "value" is assumed to be whatever is returned when
        # sampling from the model at this node. it can be a single
        # value or a dict/vector of values. in the latter case, we'll
        # generally need to override this method to do the right thing
        # with respect to whatever the model actually returns.

        if self.single_key:
            if isinstance(value, np.ndarray):
                assert(value.size == 1) # just probing to see if this is a reasonable assumption
                self.set_value(value=value[0])
            else:
                self.set_value(value=value)
        else:
            self.set_dict(value)

    def _update_mutable_cache(self):
        self._mutable_dimension = sum(self._mutable.itervalues())
        self._mutable_keys =  [k for k in self.keys() if self._mutable[k]]


    def fix_value(self, key=None):
        # hold constant (i.e., do not allow to be set or resampled) the value
        # of a particular key at this node, or of all keys (if no key is specified)

        if key is None:
            self._mutable = {k : False for k in self._mutable.iterkeys() }
        else:
            self._mutable[key] = False
        self._fixed = not any(self._mutable.itervalues())
        self._update_mutable_cache()

    def unfix_value(self, key=None):
        if key is None:
            self._mutable = {k : True for k in self._mutable.iterkeys() }
        else:
            self._mutable[key] = True
        self._fixed = False
        self._update_mutable_cache()

    def _parent_values(self):
        # return a dict of all keys provided by parent nodes, and their values
        for key in self.parent_keys_removed:
            try:
                del self._pv_cache[key]
            except KeyError:
                pass

        del self.parent_keys_removed
        self.parent_keys_removed = set()
        for (key, node) in self.parent_keys_changed:
            self._pv_cache[key] = node.get_value(key)
        del self.parent_keys_changed
        self.parent_keys_changed = set()

        for node in self.parent_nodes_added:
            self._pv_cache.update(node.get_dict())

        del self.parent_nodes_added
        self.parent_nodes_added = set()

        return self._pv_cache

    def upwards_message_normalizer(self, parent_values=None):
        # assume the generic case is that this node holds a value, modeled as a noisy sample from a GP,
        # so we just need to pass the "observed" value up to the GP, and there's no special contribution
        # from the normalizing constant of the (this node | gp mean) likelihood since it's just
        # Gaussian and sums to 1 (thus log=0).

        if parent_values is None:
            parent_values = self._parent_values()

        v = self.get_dict()
        v = v[self.single_key] if self.single_key else self._transform_values_for_model(v)

        assert(np.isfinite(v))

        for joint_model in self.params_modeled_jointly:
            joint_model.generic_upwards_message(v=v, cond=parent_values)

        # prevent physically unreasonable tail values of template params
        if self.hack_param_constraint:
            return self.param_truncation_penalty(self.label, v, coarse_tts=self.hack_coarse_tts is not None, ttr_max=self.hack_ttr_max)
        else:
            return 0.0

    @staticmethod
    def param_truncation_penalty(param, value, coarse_tts=False, ttr_max=25.0):
        p = 0.0
        if 'mult_wiggle_std' in param:
            if value > 1:
                p = -np.exp(100*(value-1))
            if value < 0:
                p = -np.exp(100*(-value))
        if 'tt_residual' in param and (not coarse_tts) and np.abs(value) > ttr_max:
            p = -(10*(np.abs(value)-ttr_max))**4
        elif 'peak_offset' in param and value > 3.0:
            p = -(10*(value-3))**4
        elif 'coda_decay' in param and value < -6:
            p = -( 10 * -(value+6))**4

        return p

    def modeled_as_joint(self):
        if len(self.params_modeled_jointly) > 0:
            assert (self.model is None or isinstance(self.model, DummyModel))
            return True
        return False

    def joint_conditional_dist(self, parent_values=None):
        assert( self.modeled_as_joint() )
        if parent_values is None:
            parent_values = self._parent_values()


        joint_model = list(self.params_modeled_jointly)[0]
        return joint_model.conditional_dist(cond=parent_values)
        

    def log_p(self, parent_values=None, v=None):
        #  log probability of the values at this node, conditioned on all parent values

        """
        if self._fixed:
            # allow fixing nodes to values that would otherwise give -inf probability
            # for example, when constraining events to a region, we still allow
            # fixed events outside of that region
            return 0.0

            # WARNING: this is commented out because it screws with event observations.
            If we treat an event's location as unknown, but fix a noisy observation of that
            event, we very much *do* want to compute the logp of that noisy observation. 
            So I need a different hack here. 

        """

        if parent_values is None:
            parent_values = self._parent_values()

        if self.modeled_as_joint():
            from sigvisa.graph.dag import ParentConditionalNotDefined
            raise ParentConditionalNotDefined()

        if v is None:
            v = self.get_dict()
            v = v[self.single_key] if self.single_key else self._transform_values_for_model(v)
        lp = self.model.log_p(x = v, cond=parent_values, key_prefix=self.key_prefix)

        # prevent physically unreasonable tail values of template params
        if self.hack_param_constraint:
            lp += self.param_truncation_penalty(self.label, v, coarse_tts=self.hack_coarse_tts is not None, ttr_max=self.hack_ttr_max)

            if "amp_transfer" in self.label and self.hack_ampt_z_max is not None:
                if not isinstance(self.model, DummyModel):
                    pred = self.model.predict(cond=parent_values)
                    std =  np.sqrt(self.model.variance(cond=parent_values, include_obs=True))
                    z = np.abs( (v - pred) / std)
                    if z > self.hack_ampt_z_max:
                        penalty = -( 10 * -(z-self.hack_ampt_z_max))**4
                        lp += penalty

        if np.isnan(lp):
            raise Exception('invalid log prob %f for value %s at node %s' % (lp, self.get_value(), self.label))

        return lp

    def _transform_values_for_model(self, vals):
        return vals

    def deriv_log_p(self, key=None, parent_values=None, parent_key=None, parent_idx=None,  **kwargs):
        # derivative of the log probability at this node, with respect
        # to a key at this node, or with respect to a key provided by
        # a parent.
        if parent_values is None:
            parent_values = self._parent_values()

        v = self.get_dict()
        if self.single_key:
            v = v[self.single_key]
            return self.model.deriv_log_p(x = v, cond=parent_values, idx=None, cond_key=parent_key, cond_idx=parent_idx, key_prefix=self.key_prefix,  **kwargs)
        else:
            return self.model.deriv_log_p(x = v, cond=parent_values, idx=key, cond_key=parent_key, cond_idx=parent_idx, key_prefix=self.key_prefix, **kwargs)


    # computes the gradient of the joint log_p with respect to the mutable values at this node.
    # this involves computing, for each key at this node, first the derivative of this node's logp, then the
    # derivatives of each stochastic child node's logp, and summing those all up.

    # for efficiency, we frame this computation as an update to a
    # preallocated array, where this node's variables start at index
    # i.

    # this is called by sigvisa.graph.dag.log_p_grad, and used to be
    # part of that method, but was factored out to allow nodes to
    # re-implement their own optimized gradient computations
    def update_mutable_grad(self, grad, i, eps, initial_lp):
        keys = self.mutable_keys()
        for (ni, key) in enumerate(keys):
            deriv = self.deriv_log_p(key=key, eps=eps[i + ni], lp0=initial_lp[self.label])

            # sum the derivatives of all child nodes wrt to this value, including
            # any deterministic nodes along the way
            child_list = self.get_stochastic_children()
            for (child, intermediate_nodes) in child_list:
                current_key = key
                d = 1.0
                for inode in intermediate_nodes:
                    d *= inode.deriv_value_wrt_parent(parent_key = current_key)
                    current_key = inode.label
                    d *= self.deriv_log_p(parent_key = current_key,
                                          eps=eps[i + ni],
                                          lp0=initial_lp[child.label])
                    deriv += d

            grad[i + ni] = deriv
        return len(keys)

    def parent_sample(self, parent_values=None, set_new_value=True):
        # sample a new value at this node conditioned on its parents
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()

        if self.model is not None:
            
            # reject samples until we meet the hacked hard constraints
            while True:
                nv = self.model.sample(cond=parent_values)
                if self.hack_param_constraint:
                    penalty = self.param_truncation_penalty(self.label, nv, coarse_tts=self.hack_coarse_tts is not None, ttr_max=self.hack_ttr_max)
                    if penalty > -1.0:
                        break
                else:
                    break

        else:
            if self.modeled_as_joint():
                joint_model = list(self.params_modeled_jointly)[0]
                nv = joint_model.sample_init(cond=self._parent_values())
            else:

                from sigvisa.graph.dag import ParentConditionalNotDefined
                raise ParentConditionalNotDefined()
        
        if set_new_value:
            self._set_values_from_model(nv)
        else:
            return nv

    def parent_predict(self, parent_values=None, set_new_value=True):
        # predict a new value at this node conditioned on its parents.
        # the meaning of "predict" varies with the model, but is
        # usually the mean or mode of the conditional distribution.
        if self._fixed: return
        if parent_values is None:
            parent_values = self._parent_values()

        if self.model is None and len(self.params_modeled_jointly) > 0:
            nv, _ = list(self.params_modeled_jointly)[0].prior()
        else:
            nv = self.model.predict(cond=parent_values)

        if set_new_value:
            self._set_values_from_model(nv)
        else:
            return nv

    def get_children(self):
        return self.children

    def set_mark(self, v=1):
        self.mark = v

    def clear_mark(self):
        self.mark = 0

    def get_mark(self):
        return self.mark

    # use custom getstate() and setstate() methods to avoid pickling
    # param models when we pickle a graph object (since these models
    # can be large, and GP models can't be directly pickled anyway).

    def __getattr__(self, name):
        if name=="model":
            if self.modelid is None:
                model = DummyModel()
            else:
                #print "loading model", self.modelid
                model = load_modelid(modelid=self.modelid, gpmodel_build_trees=False)
            self.model = model
            return self.model
        raise AttributeError("node class has no attribute %s" % name)


    def __getstate__(self):

        # ground out parent_keys_changed, etc into self._pv_cache.
        # this avoids recursive node pointers that cause pickle
        # to hit the depth limit. 
        self._parent_values()


        d = copy.copy(self.__dict__)

        try:
            self.modelid
            del d['model']
        except:
            pass

        d['parents'] = dict()

        return d

    def __setstate__(self, state):
        """
        if "model" not in state:
            if state['modelid'] is None:
                state['model'] = DummyModel()
            else:
                
                state["model"] = load_modelid(modelid=state['modelid'], gpmodel_build_trees=False)
        """
        self.__dict__.update(state)

    def mutable_dimension(self):
        # return the number of values at this node that are not fixed
        # (and can thus be optimized over, etc).
        return self._mutable_dimension

    def mutable_keys(self):
        # return the set of keys at this node whose values are not
        # fixed (and can thus be optimized over, etc).
        return self._mutable_keys

    def get_mutable_values(self):
        return [self._dict[k] for k in self._mutable_keys]

    def set_mutable_values(self, values):
        assert(len(values) == self.mutable_dimension())
        for (i,k) in enumerate(self._mutable_keys):
            self._dict[k] = values[i]
            for child in self.children:
                child.parent_keys_changed.add((k, self))

    def low_bounds(self):
        return [self._low_bounds[k] for k in self.keys() if self._mutable[k]]

    def high_bounds(self):
        return [self._high_bounds[k] for k in self.keys() if self._mutable[k]]


    def __str__(self):
        return "<NODE %s>" % self.label

    def __repr__(self):
        return self.__str__()

    #def __hash__(self):
    #    try:
    #        h =  hash(self.label)
    #    except AttributeError:
    #        h = 0
    #    return h


class DeterministicNode(Node):

    def deterministic(self):
        return True

    def parent_sample(self, parent_values=None):
        self.compute_value(parent_values=parent_values)

    def parent_predict(self, parent_values=None):
        self.compute_value(parent_values=parent_values)

    def compute_value(self, parent_values=None):
        raise NotImplementedError("compute_value method not implemented at this node!")

    def invert(self, value, parent_key, parent_values=None):
        raise NotImplementedError("invert method not implemented at this node!")

    def default_parent_key():
        raise NotImplementedError("default_parent_key method not implemented at this node!")

    def log_p(self, value=None, parent_values=None):
        raise AttributeError("cannot compute log_p for a deterministic node! %s" % (self.label))

    def deriv_log_p(**kwargs):
        raise AttributeError("cannot compute deriv_log_p for a deterministic node!")

    def set_value(self, value, key=None, parent_key=None):
        if not parent_key:
            parent_key = self.default_parent_key()

        parent_val = self.invert(value=value, parent_key=parent_key)
        self.parents[parent_key].set_value(value=parent_val, key=parent_key)
        self.compute_value()
        for child in self.children:
            child.parent_keys_changed.add((self.single_key, self))

    def get_mutable_values(self, **kwargs):
        raise AttributeError("deterministic node has no mutable values!")

    def set_mutable_values(self, **kwargs):
        raise AttributeError("deterministic node has no mutable values!")

    def fix_value(self, **kwargs):
        raise AttributeError("cannot fix/unfix values for a deterministic node!")

    def unfix_value(self, **kwargs):
        raise AttributeError("cannot fix/unfix values for a deterministic node!")
