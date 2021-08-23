"""batched collective operations for overhead amortization and better
bandwidth utilization"""

import itertools
import math
from typing import List

import torch
from torch import Tensor
import torch.distributed
from torch.distributed import ProcessGroup
import torch.nn.functional

from deepspeed.utils import instrument_w_nvtx


@instrument_w_nvtx
def reduce_scatter_coalesced(
        tensors: List[Tensor],
        group: ProcessGroup = None,
) -> List[Tensor]:
    """simultaneously reduce-scatter a list of tensors - this can be done more
    efficiently than individual reduce scatter calls

    TODO. see if PyTorch team wants a c++ verson of this for ProcessGroupNCCL,
    also maybe a custom kernel for this could help avoid a lot of the data movement
    below.
    """
    this_rank = torch.distributed.get_rank(group)
    world_sz = torch.distributed.get_world_size(group)

    tensor_partitions = [torch.chunk(tensor.view(-1), world_sz) for tensor in tensors]

    # interleave tensor data such that the correct reduced partitions of each tensor
    # end up at each rank
    tensor_partition_lst_for_each_rank: List[List[Tensor]] = [[None] * len(tensors)
                                                              for _ in range(world_sz)]
    for rank in range(world_sz):
        for tensor_idx, tensor in enumerate(tensors):
            tensor_chunk = tensor_partitions[tensor_idx][rank]

            # the tensor size isn't always evenly divisible by world size - in these
            # cases the partitions for later ranks need to be padded
            n_to_pad = math.ceil(tensor.numel() / world_sz) - tensor_chunk.numel()
            if n_to_pad:
                tensor_chunk = torch.nn.functional.pad(tensor_chunk, (0, n_to_pad))
            tensor_partition_lst_for_each_rank[rank][tensor_idx] = tensor_chunk

    tensor_partitions_for_each_rank_buffer = instrument_w_nvtx(torch.cat)(tuple(
        itertools.chain.from_iterable(tensor_partition_lst_for_each_rank)))
    tensor_partitions_for_each_rank_buffer.div_(world_sz)
    tensor_partitions_for_each_rank = list(
        torch.chunk(tensor_partitions_for_each_rank_buffer,
                    world_sz))

    # batched reduce-scatter call
    # TODO. try using _reduce_scatter_base
    instrument_w_nvtx(torch.distributed.reduce_scatter)(
        tensor_partitions_for_each_rank[this_rank],
        tensor_partitions_for_each_rank,
        group=group,
    )

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx, tensor in enumerate(tensors):
        output_lst[tensor_idx] = tensor_partitions_for_each_rank[this_rank].narrow(
            0,
            offset,
            tensor_partitions[tensor_idx][this_rank].numel())

        offset += math.ceil(tensor.numel() / world_sz)

    return output_lst
