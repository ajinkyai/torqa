import argparse
import memn2n.trainer as trainer

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="data/raw/")
    parser.add_argument("--task", type=int, default=6)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--decay_interval", type=int, default=20)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)
    parser.add_argument("--embedding_dim", type=int, default=20)
    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)

if __name__ == "__main__":
    config = parse_config()
    main(config)
