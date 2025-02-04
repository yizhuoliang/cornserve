import os
import torch.distributed as dist

def main():
    # Get WORLD_SIZE, MASTER_ADDR, and MASTER_PORT from the environment
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # Compute RANK from the POD_NAME if available; otherwise, use RANK directly.
    pod_name = os.environ.get("POD_NAME")
    if pod_name:
        try:
            # Assumes pod name is of the form "sidecar-<ordinal>"
            rank = int(pod_name.split('-')[-1])
        except ValueError:
            rank = 0
    else:
        rank = int(os.environ.get("RANK", 0))

    print(f"Starting process with rank {rank} of {world_size}.")
    print(f"Connecting to master at {master_addr}:{master_port}")
    
    # Construct the initialization URL
    init_url = f"tcp://{master_addr}:{master_port}"
    
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",
        init_method=init_url,
        rank=rank,
        world_size=world_size
    )
    
    # Now you can write your distributed training code.
    print(f"Initialized process group with rank {rank} out of {world_size}")

if __name__ == "__main__":
    main()
