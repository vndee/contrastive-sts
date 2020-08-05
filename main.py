import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contrastive Learning for Semantic Text Similarity')
    parser.add_argument('--data', '-d', type=str, default='MRPC')
    args = parser.parse_args()

    data_loader = None
    if args.data == 'MRPC':
        from loader.mrpc import load
        data_loader = load()

    print(data_loader)
