import torch
import json

def is_gpu_idle(device):
    output = torch.cuda.list_gpu_processes(device)
    output = output.split('\n')
    assert len(output) >= 2, '\n'.join(output)
    assert output[0].startswith('GPU:')

    if len(output) > 2:
        return False
    else:
        return output[1] == 'no processes are running'

def get_idle_gpus():
    gpu_list = [torch.device(f'cuda:{index}') for index in range(torch.cuda.device_count())]
    idle_gpus = [each for each in gpu_list if is_gpu_idle(each)]
    return idle_gpus

def save_cuda_memory_stats(device, path):
    if device.type != 'cuda':
        raise ValueError(f'got the wrong type of devicee: {device}')
    data = torch.cuda.memory_stats(device)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
