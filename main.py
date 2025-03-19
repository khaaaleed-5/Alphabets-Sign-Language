import argparse
from src.train import train_model
from src.eval import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pth')
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.eval:
        evaluate_model(args.checkpoint)
