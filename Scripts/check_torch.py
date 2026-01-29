import torch
print('torch version:', getattr(torch, '__version__', 'not installed'))
print('cuda available:', torch.cuda.is_available())
print('torch.version.cuda:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
    print('current device idx:', torch.cuda.current_device())
    prop = torch.cuda.get_device_properties(0)
    print('total memory (GB):', prop.total_memory / (1024**3))
