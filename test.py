import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
    else:
        print("CUDA is not available.")

check_cuda()
