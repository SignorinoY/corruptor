from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.har import HARDataModule
from model.classfier import HARLSTMClassfier, HARBiLSTMClassfier


def main():
    pl.seed_everything(10086)

    # args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HARLSTMClassfier.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    data = HARDataModule(
        args.data_dir, args.batch_size, args.num_workers, args.pin_memory
    )
    data.prepare_data()
    data.setup()

    # model
    if args.model == "lstm":
        model = HARLSTMClassfier(args.learning_rate)
    elif args.model == "bilstm":
        model = HARBiLSTMClassfier(args.learning_rate)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="har-" + args.model + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
    )
    logger = TensorBoardLogger(
        save_dir=args.log_dir, name=args.model.upper(), log_graph=True)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    # training
    trainer.fit(model, datamodule=data)

    # testing
    trainer.test(datamodule=data)


if __name__ == "__main__":
    main()
