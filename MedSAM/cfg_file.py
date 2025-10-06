import argparse

def parse_args():
      parse = argparse.ArgumentParser(description='SAM_XCT')
      parse.add_argument('-mod', type=str, default='sam_lora', help='mod type:sam_lora,sam_adpt,val_ad')
      parse.add_argument('-mid_dim', type=int, default=4 , help='middle dim of adapter or the rank of lora matrix')
      parse.add_argument('-thd', type=bool, default=False , help='3d or not')
      parse.add_argument('-chunk', type=int, default=None , help='crop volume depth')
      parse.add_argument('--exp', type=str, default='zm_p2', dest='exp',
                     help='exp_name ') 
      parse.add_argument('--date', type=str, default='MASA') 
      parse.add_argument('--geometry', type=str, default='2540geometry.xml') 
      parse.add_argument('--batch_size', type=int, default=4, dest='batch_size',
                     help='batch_size')
      parse.add_argument('--epoch', type=int, default=200, dest='epoch',
                     help='epoch')
      parse.add_argument('--lr', type=float, default=0.0002, dest='lr',
                     help='lr') 
      parse.add_argument('--gpu', type=str, default='2', dest='gpuid',
                     help='gpu is split by')
      parse.add_argument('--experiment_path', type=str, default='/hdd2/wjj/SAMXct/')    
      parse.add_argument('--phase',type=str,default='train')                    
      parse.add_argument('--arch', type=str, default='SAM_XCT', dest='arch',
                     help='architecture of network')
      parse.add_argument('--print_freq', type=int, default=1, dest='print_freq',
                     help='print freq')
      parse.add_argument('--resume', type=str, default='final', dest='resume',
                     help='resume model')
      parse.add_argument('--num_views', type=int, default=1, dest='num_views',
                     help='none')
      parse.add_argument('--output_channel', type=int, default=128, dest='output_channel',
                     help='output_channel')
      parse.add_argument('--classes', type=int, default=2, dest='classes',
                     help='output_classes')
      parse.add_argument('--loss', type=str, default='l2', dest='loss',
                     help='loss')
      parse.add_argument('--optim', type=str, default='adam', dest='optim',
                     help='optim')
      parse.add_argument('--weight_decay', type=float, default=0, dest='weight_decay',
                     help='weight_decay')
      parse.add_argument('--init_gain', type=float, default=0.02, dest='init_gain',
                     help='init_gain')
      parse.add_argument('--init_type', type=str, default='standard', dest='init_type',
                     help='init_type')
      parse.add_argument('--save_epoch_freq', type=int, default=1, dest='save_epoch_freq',
                     help='save_epoch_freq')
      parse.add_argument('--model_name', type=str, default='best_model', dest='model_name',
                      help='model_name')
      parse.add_argument('--contrain', type=str, default=None, dest='contrain',
                      help='contrain')
      parse.add_argument("--device", type=str, default="cuda:0"),
      parse.add_argument("-chk","--checkpoint",type=str,default="MedSAM/work_dir/MedSAM/medsam_vit_b.pth",help="path to the trained model"),
      args = parse.parse_args()
      return args