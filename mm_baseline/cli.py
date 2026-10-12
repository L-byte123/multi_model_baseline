import argparse


def main():
    parser = argparse.ArgumentParser(description="STFT + operating-text fault diagnosis baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate numeric spectrograms and real PNG images")
    prepare.add_argument("--recordings", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--window", type=int, default=2048)
    prepare.add_argument("--step", type=int, default=2048)
    prepare.add_argument("--n-fft", type=int, default=256)
    prepare.add_argument("--hop", type=int, default=64)
    prepare.add_argument("--fmin", type=float, default=0)
    prepare.add_argument("--fmax", type=float)
    prepare.add_argument("--reference-preprocessing", help="Training preprocessing.json for new inference signals")

    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", default="configs/passt_text.json")
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="auto")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    prediction.add_argument("--device", default="auto")
    demo = commands.add_parser("demo", help="Create synthetic raw recordings, not a real benchmark")
    demo.add_argument("--output", default="data/demo_raw")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .preprocess import prepare
        print(prepare(**args))
    elif command == "train":
        from .engine import train
        args["config_path"] = args.pop("config")
        args["device_name"] = args.pop("device")
        train(**args)
    elif command == "predict":
        from .engine import predict
        args["checkpoint_path"] = args.pop("checkpoint")
        args["device_name"] = args.pop("device")
        predict(**args)
    else:
        from .demo import make_demo
        print(make_demo(**args))


if __name__ == "__main__":
    main()
