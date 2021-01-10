from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Inpainting/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--input_dir_mask', help='input mask dir', default='Input/Inpainting/Masks')
    parser.add_argument('--input_name_mask', help='input mask name', required=True)
    parser.add_argument('--inpainting_start_scale', help='inpaiting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            mask = functions.read_image_mask(opt)
            mask = functions.adjust_scales2image(mask, opt)
            if mask.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(real, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)




