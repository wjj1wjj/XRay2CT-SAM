import argparse
from MedXct_trainer import Trainer_XctNet
from multiDataset import offline_multiDataSet
from OurFn import our_gan
import tensorboardX as tbX
import torch 
import time
import os
import shutil
import cfg_file


if __name__ == '__main__':
  
  args = cfg_file.parse_args()
  # check gpu
  assert (torch.cuda.is_available())
  split_gpu = str(args.gpuid).split(',')
  args.gpu_ids = [int(i) for i in split_gpu]
  
  args.fine_size=128
  args.output_path = os.path.join(args.experiment_path ,args.exp ,args.date,'model')
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
    
  log_dir=os.path.join(args.experiment_path ,args.exp ,args.date,'loss')
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
  args.dataroot=os.path.join('/hdd2/wjj',args.exp,'h5py')
  print(args.dataroot)  
  #已修改
  train_datasetfile = os.path.join("/hdd2/wjj/experiment/",'train.txt')  
  valid_datasetfile =os.path.join("/hdd2/wjj/experiment/",'val.txt')
  
  train_dataset = offline_multiDataSet(args,train_datasetfile)
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=1,
    pin_memory=True,
    collate_fn=our_gan,
    drop_last=True)
  
  print('total training images: %d' % (len(train_dataloader)*args.batch_size))

  # valid dataset
  valid_dataset = offline_multiDataSet(args, valid_datasetfile)
  valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    collate_fn=our_gan,
    drop_last=True)
  print('total validation images: %d' % len(valid_dataloader))

  #get model
  model=Trainer_XctNet(args)
  start_epoch = 1
  if args.contrain is not None:
    start_epoch = model.load()+1

  tb = tbX.SummaryWriter(log_dir=log_dir)
  for epoch in range(start_epoch, args.epoch):
    #train
    start_time = time.time()
    train_recon_loss,train_seg_loss,train_loss=model.train_epoch(train_loader=train_dataloader,epoch=epoch,tb=tb)
    train_end_time = time.time()
    tb.add_scalar('Train_Recon_Loss', train_recon_loss, epoch)
    tb.add_scalar('Train_Seg_Loss', train_seg_loss, epoch)
    tb.add_scalar('Train_Loss', train_loss, epoch)
    

    #val
    val_recon_loss,val_seg_loss,val_loss=model.validate(val_loader=valid_dataloader)
    tb.add_scalar('Val_Recon_Loss', val_recon_loss, epoch)
    tb.add_scalar('Val_Seg_Loss', val_seg_loss, epoch)
    tb.add_scalar('Val_Loss', val_loss, epoch)

    # save curr model
    print('saving the model at the end of epoch %d' %epoch)
    model.save(val_recon_loss, epoch=epoch)
    # save model several epoch
    if epoch%40==0:
      model.save_epoch(val_recon_loss, epoch=epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec \t' %
          (epoch, args.epoch, time.time() - start_time))