1. Настройка логгирования


pip install \
  torch==2.2.2 \
  torchaudio==2.2.2 \
  torchvision==0.17.2



 python train.py   datasets=onebatchtest   dataloader.batch_size=1   dataloader.num_workers=0   trainer.n_epochs=50   trainer.epoch_len=1   writer.mode=online   writer.run_name=onebatch_overfit_b6