import os
import json
import argparse
import torch
import deepspeed
from torch import nn
from torch.utils.data.distributed import DistributedSampler


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False, zero=0):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        mlp = [self.linear]
        mlp.append(torch.nn.Linear(hidden_dim, hidden_dim//2))
        for _ in range(6):
            l = torch.nn.Linear(hidden_dim//2, hidden_dim//2)
            mlp.append(l)
        mlp.append(torch.nn.Linear(hidden_dim//2, hidden_dim))
        l = torch.nn.Linear(hidden_dim, hidden_dim)
        l.weight = self.linear.weight
        l.bias = self.linear.bias
        mlp.append(l)
        if zero == 3:
            deepspeed.zero.register_external_parameter(self, self.linear.weight)
            deepspeed.zero.register_external_parameter(self, self.linear.bias)
        self.mlp = nn.Sequential(*mlp)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.mlp(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=0)
    parser.add_argument('--contiguous-gradients', default=False, type=bool)
    parser.add_argument('--reduce-scatter', default=True, type=bool)
    parser.add_argument('--hidden-size', default=16, type=int)
    parser.add_argument('--tmp', default='/tmp', help="temporary directory to save intermediate data")
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    config_dict["zero_optimization"]['contiguous_gradients'] = args.contiguous_gradients
    config_dict["zero_optimization"]["reduce_scatter"] = args.reduce_scatter
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)


rank = int(os.environ['RANK'])
print('seed:', 2222 + rank)
torch.random.manual_seed(2222 + rank)

config_dict = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 1,
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.02,
            "weight_decay": 0.01,
            "bias_correction": True,
            "eps": 1e-6
        }
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True,
        "initial_scale_power": 10
    },
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
        "reduce_scatter": True,
        "contiguous_gradients": False,
        "reduce_bucket_size": 20,
        "stage3_param_persistence_threshold": 1
    }
}
#        "initial_scale_power": 15
args = get_args('/tmp/', config_dict)
hidden_dim = args.hidden_size

model = SimpleModel(hidden_dim, empty_grad=False, zero=args.zero)

model, _, _,_ = deepspeed.initialize(args=args,
                                     model=model,
                                     model_parameters=model.parameters(),
                                     dist_init_required=True)


def print_params(tag, model):
    if torch.distributed.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))


data_loader = get_data_loader(model=model,
                              total_samples=1000,
                              hidden_dim=hidden_dim,
                              device=model.device)

def _get_log_filename(args):
    return f"/tmp/loss_log_stage{args.zero}" \
            f".h{args.hidden_size}" \
            f".cg{args.contiguous_gradients}"\
            f".rc{args.reduce_scatter}.txt"

with open(_get_log_filename(args), 'w') as log_file:
    #print_params('pre-train', model)
    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        #if torch.distributed.get_rank() == 0 and model.is_gradient_accumulation_boundary():
        model.backward(loss)
        model.step()
        if torch.distributed.get_rank() == 0 and model.is_gradient_accumulation_boundary():
            print("{}, LOSS: {}".format(n, loss.item()))
            log_file.write(f'{n},{loss.item()}\n')
        #print_params('step={}'.format(n), model)
        if n == 20: break
