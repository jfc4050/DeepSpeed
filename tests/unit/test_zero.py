import math
from typing import Dict, List, Set
import pytest
import torch.distributed as dist
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.loss import L1Loss
from torch.nn.parameter import Parameter

from .common import distributed_test
from .simple_model import SimpleModel, random_dataloader, args_from_dict

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def run_unbalanced_gradients(model, data_loader):
    def drop_some_gradients(model, iter):
        odd_iteration = iter % 2
        for i, p in enumerate(model.parameters()):
            p.requires_grad = (i % 2) == odd_iteration

    def enable_grads(model):
        for p in model.parameters():
            p.requires_grad = True

    for i, batch in enumerate(data_loader):
        drop_some_gradients(model, i + 1)
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()
        enable_grads(model)


@pytest.mark.parametrize('zero_stage', [1, 2, 3])
def test_zero_unbalanced_gradients(tmpdir, zero_stage):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_zero_unbalanced_gradients(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        run_unbalanced_gradients(model, data_loader)

    _test_zero_unbalanced_gradients(args=args, model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
@pytest.mark.parametrize('zero_stage', [3])
def test_zero3_repeat_forward_loop(tmpdir, zero_stage):

    # force all params to be partitioned by forcing threshold=0
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 0
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    class AlbertLikeModel(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, y):
            # run the same layer multiple times in a loop - to test a stack of forwards, followed by a stack of backwards
            hidden = x
            for i in range(3):
                hidden = hidden + self.linear(hidden)
            return self.cross_entropy_loss(hidden, y)

    model = AlbertLikeModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_zero3_repeat_forward_loop(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero3_repeat_forward_loop(args=args, model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32(tmpdir, zero_stage):

    # TODO:
    # - need to test with multiple param groups

    # force all params to be partitioned by forcing threshold=0
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 0
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    @distributed_test(world_size=[2])
    def _test_zero_to_fp32():
        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim, n_layers):
                super().__init__()
                self.ll = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim,
                                    hidden_dim) for i in range(n_layers))
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 2

        world_size = dist.get_world_size()
        # we want at least 2x layers as there are gpus to trigger round_robin_fp16_groups reshuffle in zero2
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers)

        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        def dump_state_dict(model):
            if dist.get_rank() != 0:
                return
            for name, param in model.named_parameters():
                print(f"{name} {param}")

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()
        print(orig_state_dict)

        fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
        #dump_state_dict(fp32_model)

        fp32_state_dict = fp32_model.state_dict()
        for name in orig_state_dict.keys():
            # float() workaround for torch<1.6
            assert torch.allclose(orig_state_dict[name].float(),
                                  fp32_state_dict[name].float())

    _test_zero_to_fp32()


def _ds_initialize_for_param_partitioning_testing(model: Module,
                                                  cfg: dict) -> DeepSpeedEngine:
    ds_engine, _, _, _ = deepspeed.initialize(
        config=cfg,
        model=model,
        model_parameters=model.parameters()
    )

    return ds_engine


def _print_with_rank(msg: str) -> None:
    print(f"RANK{dist.get_rank()}: {msg}")


def _assert_partition_status(model: Module,
                             valid_statuses: Set[ZeroParamStatus]) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status in valid_statuses, param.ds_summary()


def _assert_fully_available(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status == ZeroParamStatus.AVAILABLE


class EltwiseMultiplicationModule(Module):
    def __init__(self, weight: Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        _assert_fully_available(self)
        result = self.weight * x

        return result


class EltwiseMultiplicationTestNetwork(Module):
    """used for testing purposes"""
    def __init__(
            self,
            weight1: Parameter,
            weight2: Parameter,
            weight3: Parameter,
    ) -> None:
        super().__init__()
        self.__layer1 = EltwiseMultiplicationModule(weight1)
        self.__layer2 = EltwiseMultiplicationModule(weight2)
        self.__layer3 = EltwiseMultiplicationModule(weight3)

        self.loss = L1Loss(reduction="none")

    def forward(self, x: Tensor, y: Tensor, prefetching: bool) -> Dict[str, Tensor]:
        _assert_partition_status(
            self,
            {
                ZeroParamStatus.NOT_AVAILABLE,
                ZeroParamStatus.INFLIGHT,
                ZeroParamStatus.AVAILABLE
            } if prefetching else {ZeroParamStatus.NOT_AVAILABLE})

        _assert_partition_status(
            self.__layer1,
            {ZeroParamStatus.INFLIGHT if prefetching else ZeroParamStatus.NOT_AVAILABLE})
        hidden1 = self.__layer1(x)
        _assert_partition_status(self.__layer1, {ZeroParamStatus.NOT_AVAILABLE})

        _assert_partition_status(self.__layer2,
                                 {
                                     ZeroParamStatus.AVAILABLE
                                     if prefetching else ZeroParamStatus.NOT_AVAILABLE
                                 })
        hidden2 = self.__layer2(hidden1)
        _assert_partition_status(self.__layer2, {ZeroParamStatus.NOT_AVAILABLE})

        _assert_partition_status(self.__layer3,
                                 {
                                     ZeroParamStatus.AVAILABLE
                                     if prefetching else ZeroParamStatus.NOT_AVAILABLE
                                 })
        y_hat = self.__layer3(hidden2)
        _assert_partition_status(self.__layer3,
                                 {
                                     ZeroParamStatus.AVAILABLE
                                     if prefetching else ZeroParamStatus.NOT_AVAILABLE
                                 })

        loss = self.loss(y_hat, y)

        _assert_partition_status(
            self,
            {
                ZeroParamStatus.NOT_AVAILABLE,
                ZeroParamStatus.INFLIGHT,
                ZeroParamStatus.AVAILABLE
            } if prefetching else {ZeroParamStatus.NOT_AVAILABLE})

        return {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "y_hat": y_hat,
            "loss": loss,
        }


@pytest.mark.parametrize("param_persistence_threshold", [0, 10])
@pytest.mark.parametrize("fp16_enabled", [True, False])
@pytest.mark.parametrize("iteration", list(range(25)))
def test_zero3_param_partitioning_base(
        param_persistence_threshold: int,
        fp16_enabled: bool,
        iteration: int,
) -> None:
    @distributed_test(world_size=[2])
    def _test_zero3_param_partitioning():
        m = 3
        n = 5
        weights = [
            Parameter(torch.full((m,
                                  n),
                                 i * (1 + dist.get_rank()),
                                 dtype=torch.float32)) for i in range(3)
        ]
        model = EltwiseMultiplicationTestNetwork(*weights)

        ds_engine = _ds_initialize_for_param_partitioning_testing(
            model,
            {
                "train_micro_batch_size_per_gpu": 1,
                "zero_optimization": {
                    "stage": 3,
                    "stage3_max_reuse_distance": 0,
                    "stage3_param_persistence_threshold": param_persistence_threshold,
                },
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 1.
                    }
                },
                "fp16": {
                    "enabled": fp16_enabled,
                    "loss_scale": 1.,
                }
            })
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(weight.ds_tensor.data,
                                                    (i + 1) * (1 + dist.get_rank()))

        def create_tensor(vals):
            return torch.as_tensor(
                vals,
                dtype=torch.float16 if fp16_enabled else torch.float32,
                device=ds_engine.device)

        expected_hidden1 = create_tensor([
            [1,
             1,
             1,
             1,
             1],
            [1,
             1,
             1,
             2,
             2],
            [2,
             2,
             2,
             2,
             2],
        ])
        expected_hidden2 = create_tensor([
            [2,
             2,
             2,
             2,
             2],
            [2,
             2,
             2,
             8,
             8],
            [8,
             8,
             8,
             8,
             8],
        ])
        expected_yhat = create_tensor([[6,
                                        6,
                                        6,
                                        6,
                                        6],
                                       [6,
                                        6,
                                        6,
                                        48,
                                        48],
                                       [48,
                                        48,
                                        48,
                                        48,
                                        48]])
        expected_loss = create_tensor([
            [5,
             5,
             5,
             5,
             5],
            [5,
             5,
             5,
             47,
             47],
            [47,
             47,
             47,
             47,
             47],
        ])

        for train_iter in range(3):
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
            activations = ds_engine(
                x=torch.ones((m,
                              n),
                             dtype=torch.float16,
                             device=ds_engine.device),
                y=torch.ones((m,
                              n),
                             dtype=torch.float16,
                             device=ds_engine.device),
                prefetching=train_iter > 0,
            )
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradients: List[Tensor] = avgd_gradients[0]

            # FIXME: flaky failures on [fp16, persist_thresh]
            # just taking some notes on different race conditions encountered
            # with multiple runs...
            #
            # A: 2x [F-10]
            #    3x [F-0]
            #    1x [T-0]

            dloss_wrt_layer1, dloss_wrt_layer2, dloss_wrt_layer3 = weight_gradients
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x
            if dist.get_rank() == 0:
                assert torch.allclose(dloss_wrt_layer3, create_tensor([2] * 8)), dloss_wrt_layer3
                assert torch.allclose(dloss_wrt_layer2, create_tensor([3 * 1] * 8)), dloss_wrt_layer2
                # FIXME: A - get 4.5 - ??? (note that world size is 2)
                assert torch.allclose(dloss_wrt_layer1, create_tensor([3 * 2 * 1] * 8)), dloss_wrt_layer1
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(dloss_wrt_layer3, create_tensor(([8] * 7) + [0])), dloss_wrt_layer3
                assert torch.allclose(dloss_wrt_layer2,
                                      create_tensor(([6 * 2] * 7) + [0])), dloss_wrt_layer1
                assert torch.allclose(dloss_wrt_layer1,
                                      create_tensor(([6 * 4 * 1] * 7) + [0])), dloss_wrt_layer1
            else:
                raise RuntimeError("test has world size of two")

    _test_zero3_param_partitioning()


@pytest.mark.parametrize("world_sz", [1, 2, 4])
@pytest.mark.parametrize("param_sz", [8100])
def test_zero3_param_partitioning_large_param(world_sz: int, param_sz: int):
    class LargeParamModel(Module):
        def __init__(self):
            super().__init__()
            self.param = Parameter(torch.empty(param_sz, dtype=torch.float32))

        def forward(self, x: Tensor) -> Tensor:
            return x * self.param

    @distributed_test(world_size=[world_sz])
    def _distributed_test():
        model = LargeParamModel()
        ds_engine = _ds_initialize_for_param_partitioning_testing(
            model,
            {
                "train_micro_batch_size_per_gpu": 1,
                "zero_optimization": {
                    "stage": 3,
                    "stage3_max_reuse_distance": 0,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                },
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 1.
                    }
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 1.,
                }
            })

        model.param.ds_tensor.data = torch.full_like(model.param.ds_tensor.data,
                                                     dist.get_rank())

        for _ in range(3):  # test multiple iterations to cover prefetching
            activation: Tensor = ds_engine(
                torch.ones(param_sz,
                           dtype=torch.float16,
                           device=ds_engine.device))

            partition_sz = math.ceil(param_sz / world_sz)
            for rank_idx, start_idx in enumerate(range(0, param_sz, partition_sz)):
                activation_from_partition = activation[start_idx:start_idx +
                                                       partition_sz]
                assert torch.allclose(
                    activation_from_partition,
                    torch.full_like(activation_from_partition,
                                    rank_idx))

            ds_engine.backward(activation.sum())

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradient, = avgd_gradients[0]

            assert torch.allclose(weight_gradient, torch.full_like(weight_gradient, 1))

    _distributed_test()


@pytest.mark.parametrize("world_sz", [1, 2, 4])
@pytest.mark.parametrize("param_sz", [100, 1_000, 10_000])
@pytest.mark.parametrize("n_layers", [100, 1_000])
def test_zero3_param_partitioning_many_params(world_sz, param_sz, n_layers):
    class ManyParamModel(Module):
        def __init__(self) -> None:
            super().__init__()

            self.modulelist = ModuleList(
                EltwiseMultiplicationModule(
                    weight=Parameter(torch.empty((param_sz,
                                                  ),
                                                 dtype=torch.float32)))
                for _ in range(n_layers))

        def forward(self, x: Tensor) -> Tensor:
            activations = []

            for module in self.modulelist:
                print(f"{dist.get_rank()}: xval: {x.shape}")
                x = module(x)
                activations.append(x)

            return activations

    @distributed_test(world_size=[world_sz])
    def _distributed_test():
        model = ManyParamModel()
        ds_engine = _ds_initialize_for_param_partitioning_testing(
            model,
            {
                "train_micro_batch_size_per_gpu": 1,
                "zero_optimization": {
                    "stage": 3,
                    "stage3_max_reuse_distance": 0,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                },
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 1.
                    }
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 1.,
                }
            })
        for layer_num in range(n_layers):
            param = model.modulelist[layer_num].weight
            param.ds_tensor.data = torch.full_like(param.ds_tensor.data,
                                                   2 * layer_num * dist.get_rank())

        for _ in range(3):  # test multiple iterations to cover prefetching
            activations: List[Tensor] = ds_engine(
                torch.ones((param_sz,
                            ),
                           dtype=torch.float16,
                           device=ds_engine.device))
            assert len(activations) == n_layers

            partition_sz = math.ceil(param_sz / world_sz)
            expected_activations = torch.empty(param_sz,
                                               dtype=torch.float16,
                                               device=ds_engine.device)
            for start_idx in range(0, param_sz, partition_sz):
                expected_activations[start_idx:start_idx +
                                     partition_sz] = dist.get_rank()

            for layer_num, activation in enumerate(activations):
                expected_activations *= 2 * layer_num
                assert torch.allclose(activation, expected_activations)

            # TODO. finish writing this test
            ds_engine.backward(activations[-1].sum())
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradients: List[Tensor] = avgd_gradients[0]

            for layer_num, activation in enumerate(weight_gradients):
                pass

    _distributed_test()
