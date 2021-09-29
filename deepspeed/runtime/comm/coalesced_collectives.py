"""batched collective operations for overhead amortization and better
bandwidth utilization"""

import math
from typing import List

import torch
from torch import Tensor
import torch.distributed
from torch.distributed import ProcessGroup
import torch.nn.functional

from deepspeed.utils import instrument_w_nvtx


@instrument_w_nvtx
@torch.no_grad()
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
    tensor_partitions_for_each_rank: List[Tensor] = [None] * world_sz
    tensor_partition_padded_sizes = [math.ceil(t.numel() / world_sz) for t in tensors]
    for rank in range(world_sz):
        tensor_partitions_lst_with_padding = []

        for tensor_idx in range(len(tensors)):
            # add tensor content
            tensor_chunk = tensor_partitions[tensor_idx][rank]
            tensor_partitions_lst_with_padding.append(tensor_chunk)

            # add padding if necessary
            padding_sz = tensor_partition_padded_sizes[tensor_idx] - tensor_chunk.numel()
            if padding_sz > 0:
                tensor_partitions_lst_with_padding.append(
                    torch.empty(padding_sz,
                                dtype=tensor_chunk.dtype,
                                device=tensor_chunk.device))

        tensor_partitions_for_each_rank[rank] = instrument_w_nvtx(
            torch.cat)(tensor_partitions_lst_with_padding)

    # batched reduce-scatter call
    # TODO. try using _reduce_scatter_base
    instrument_w_nvtx(torch.distributed.reduce_scatter)(
        tensor_partitions_for_each_rank[this_rank],
        tensor_partitions_for_each_rank,
        group=group,
    )

    # post-divide
    tensor_partitions_for_each_rank[this_rank].div_(world_sz)

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partitions_for_each_rank[this_rank].narrow(
            0,
            offset,
            tensor_partitions[tensor_idx][this_rank].numel())

        offset += tensor_partition_padded_sizes[tensor_idx]

    return output_lst
