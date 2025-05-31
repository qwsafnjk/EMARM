import argparse
import torch
import pandas as pd
import os
from sampling import sampling_fuzzy
from inout import load_data
from time import time

# Training device
torch.cuda.set_device(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               beta: int = 32,
               seed: int = None):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: FJSSP instance.
        beta: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time()
    s, mss = sampling_fuzzy(ins, encoder, decoder, K=beta, device=dev)
    exe_time = time() - st

    results = {'NAME': ins['name'],
               'TIME': exe_time,
               'solution': mss}
    return results


parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default="your/checkpoint.pt",
                    help="Path to the model.")
parser.add_argument("-benchmark", type=str, required=False,
                    default='6*6', help="Name of the benchmark for testing.")
parser.add_argument("-beta", type=int, default=1024, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    from PointerNet import GATEncoder
    print(f"Using {dev}...")

    # Load the model
    print(f"Loading {args.model_path}")
    enc_w, dec_ = torch.load(args.model_path, map_location=dev)
    enc_ = GATEncoder(18).to(dev)   # Load weights to avoid bug with new PyG
    enc_.load_state_dict(enc_w)
    m_name = args.model_path.rsplit('/', 1)[1].split('.', 1)[0]


    path = f'your/benchmarks/{args.benchmark}'
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    out_file = f'your/output/{m_name}_{args.benchmark}-B{args.beta}_{args.seed}.csv'


    for file in os.listdir(path):
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data(os.path.join(path, file), device=dev)
        res = validation(enc_, dec_, instance,
                         beta=args.beta, seed=args.seed)

        # Save results
        pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',')
