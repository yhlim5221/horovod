# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy

import torch

from horovod.common.elastic import ObjectState
from horovod.torch.elastic.sampler import ElasticSampler
from horovod.torch.functions import allgather_object, \
    broadcast_object, broadcast_optimizer_state, broadcast_parameters
from horovod.torch.mpi_ops import rank, cross_rank, cross_size, local_rank, local_size
from horovod.common.process_sets import ProcessSet, global_process_set, \
        _temp_process_set_object, add_process_set, remove_process_set, \
        is_process_set_included

class TorchState(ObjectState):
    """State representation of a PyTorch training process.

    Multiple models and optimizers are supported by providing them as
    kwargs. During initialization, `TorchState` will assign attributes
    for every keyword argument, and handle its state synchronization.

    Args:
        model: Optional PyTorch model.
        optimizer: Optional PyTorch optimizer.
        kwargs: Attributes sync, will be exposed as attributes of the object. If a handler exists
                for the attribute type, it will be used to sync the object, otherwise it will be
                handled an ordinary Python object.
    """
    def __init__(self, model=None, optimizer=None, **kwargs):
        kwargs.update(dict(model=model, optimizer=optimizer))
        self._handlers, kwargs = _get_handlers(kwargs)
        for name, handler in self._handlers.items():
            setattr(self, name, handler.value)

        self.cross_rank = cross_rank()
        self.cross_size = cross_size()
        super(TorchState, self).__init__(bcast_object=broadcast_object,
                                         get_rank=rank,
                                         **kwargs)

    def save(self):
        for handler in self._handlers.values():
            handler.save()
        super(TorchState, self).save()

    def restore(self):
        for handler in self._handlers.values():
            handler.restore()
        super(TorchState, self).restore()

    def _sync(self, root_rank=0, process_set_id=0):
        for handler in self._handlers.values():
            handler.sync(root_rank, process_set_id)
        super(TorchState, self).sync(root_rank, process_set_id)


    def sync(self, old_rank=0, process_set_id=0):
        if self.cross_size == 1:
            self._sync(process_set_id)
            return

        worker_info = [self.cross_rank, self._rank(), old_rank]
        cluster_map = allgather_object(torch.IntTensor(worker_info), name='clustermap')
        host_map = dict()
        for idx, item in enumerate(cluster_map):
            old_new = 'old' if item[2].item() else 'new'
            complement_old_new = 'new' if old_new == 'old' else 'old'
            if item[0].item() in host_map:
                host_map[item[0].item()][old_new].append(item[1].item())
            else:
                host_map[item[0].item()] = {old_new:[item[1].item()], complement_old_new:[]}
        no_old_hosts = []
        for i in range(self.cross_size):
            if not host_map[i]['old']:
                no_old_hosts.append(host_map[i]['new'][0])
                host_map[i]['old']=[host_map[i]['new'][0]]
                host_map[i]['new'].remove(host_map[i]['new'][0])
        if len(no_old_hosts) == self.cross_size:
            return
        if no_old_hosts:
            """ If all the hosts are new, then no need to sync
                add process set of no old worker hosts' first rank with rank 0
                call _sync with them first
                remove process set
            """
            no_old_hosts.append(0)
            temp = add_process_set(no_old_hosts)
            if is_process_set_included(temp.process_set_id):
                _sync(process_set_id=temp.process_set_id)
            remove_process_set(temp)

        host_process_set_list = dict()
        for i in range(self.cross_size):
            if not host_map[i]['new']:
                continue
            print(str(i)+" "+str([host_map[i]['old'][0]]+host_map[i]['new']))
            host_process_set_list[i] = add_process_set([host_map[i]['old'][0]]+host_map[i]['new'])
        if host_process_set_list.get(self.cross_rank) and \
                is_process_set_included(host_process_set_list[self.cross_rank].process_set_id): 
            self._sync(root_rank=host_process_set_list[self.cross_rank].ranks[0],
                    process_set_id=host_process_set_list[self.cross_rank].process_set_id)
        for host_process_set in host_process_set_list.values():
            remove_process_set(host_process_set)

    def __setattr__(self, name, value):
        if hasattr(self, name) and name in self._handlers:
            self._handlers[name].set_value(value)
        super().__setattr__(name, value)


class StateHandler(object):
    def __init__(self, value):
        self.value = value

    def save(self):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()

    def set_value(self, value):
        self.value = value
        self.save()


class ModelStateHandler(StateHandler):
    def __init__(self, model):
        super().__init__(model)
        self._saved_model_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_model_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_model_state)

    def sync(self, root_rank=0, process_set_id=0):
        print("sync models...")
        broadcast_parameters(self.value.state_dict(), root_rank=root_rank, 
                process_set=_temp_process_set_object(process_set_id))


class OptimizerStateHandler(StateHandler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._saved_optimizer_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_optimizer_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_optimizer_state)

    def sync(self, root_rank=0, process_set_id=0):
        print("sync optimizer...")
        broadcast_optimizer_state(self.value, root_rank=root_rank, 
                process_set=_temp_process_set_object(process_set_id))


class SamplerStateHandler(StateHandler):
    def __init__(self, sampler):
        super().__init__(sampler)
        self._saved_sampler_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_sampler_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_sampler_state)

    def sync(self, root_rank=0, process_set_id=0):
        state_dict = self.value.state_dict()
        print("sync sampler...")
        # Broadcast and load the state to make sure we're all in sync
        self.value.load_state_dict(broadcast_object(state_dict, root_rank=root_rank,
            process_set=_temp_process_set_object(process_set_id)))


def _union(sets):
    # Union a list of sets into a single set
    return set().union(*sets)


_handler_registry = [
    (torch.nn.Module, ModelStateHandler),
    (torch.optim.Optimizer, OptimizerStateHandler),
    (ElasticSampler, SamplerStateHandler),
]


def get_handler_registry():
    return _handler_registry


def set_handler_registry(registry):
    global _handler_registry
    _handler_registry = registry


def _get_handler(v):
    for handler_type, handler_cls in _handler_registry:
        if isinstance(v, handler_type):
            return handler_cls(v)
    return None


def _get_handlers(kwargs):
    handlers = {}
    remainder = {}
    for k, v in kwargs.items():
        handler = _get_handler(v)
        if handler:
            handlers[k] = handler
        else:
            remainder[k] = v
    return handlers, remainder
