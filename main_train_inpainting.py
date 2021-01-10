from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Inpainting/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--input_dir_mask', help='input mask dir', default='Input/Inpainting/Masks')
    parser.add_argument('--input_name_mask', help='input mask name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masks = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        mask = functions.read_image_mask(opt)
        functions.adjust_scales2image(mask, opt)
        #mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
        
        train_inpainting(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate_inpainting(Gs,Zs,reals,NoiseAmp,opt)
