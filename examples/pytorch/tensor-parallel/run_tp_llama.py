import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

import os

from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

def setup(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU communication
        init_method='env://',  # Use environment variables for initialization
        rank=rank,
        world_size=world_size
    )
    # Set the current device based on the rank
    torch.cuda.set_device(rank)

def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()

def run(rank, world_size):
    """Main training/evaluation loop."""
    # Set environment variables for the distributed setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can use any available port

    setup(rank, world_size)
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel='auto', tensor_parallel_size=2)

    # Apply tensor parallel plan
    # model = AutoModelForCausalLM.from_pretrained("model name", tensor_parallel='auto')

    # Compiler-level optimization
    # model = torch.compile(model)

    # Example inference/generation
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 512), device='cuda')
    output = model(prompt_ids)

    print(output)

    # Example backward pass (for training)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # label = torch.tensor([your_labels_here], device=rank)
    # output = model(prompt_ids)
    # loss = loss_fn(output.logits, label)
    # loss.backward()

    # Generation
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Rank {rank} generated: {generated_text}")

    # Cleanup
    cleanup()

def main():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
