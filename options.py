import argparse
import os
from utils import mkdirs
from datetime import datetime


class Option:
    """This class defines options used during both training and CNN_PET_ADCN time. It also implements several helper
    functions such as parsing, printing, and saving the options. It also gathers additional options defined in
    <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and CNN_PET_ADCN."""
        # basic settings
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--model', type=str, default='CrossTransformer_IT')
        parser.add_argument('--num_workers', type=int, default=8)

        # dataset parameters
        parser.add_argument('--aug', type=bool, default=True)
        parser.add_argument('--task', type=str, default='COG')
        parser.add_argument('--mode', type=str, default='train')

        # 37 tabular features
        parser.add_argument('--tabular', nargs='+', type=str,
                            default=["age", "gender", "education", "apoe", 'trailA', 'trailB', 'boston', 'animal',
                                     'gds', 'mmse', 'npiq_DEL',	'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX',
                                     'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE',
                                     'npiq_APP', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                                     'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL',
                                     'his_CVHATT', 'his_CBSTROKE', 'his_HYPERTEN', 'his_PSYCDIS', 'his_ALCOHOL'])
        parser.add_argument('--tabular_continues_idx', nargs='+', type=int,
                            default=[0, 2, 3, 4, 5, 7])
        parser.add_argument('--tabular_categorical_idx', nargs='+', type=int,
                            default=[1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                     27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

        # 6 tabular features
        # parser.add_argument('--tabular', nargs='+', type=str,
        #                     default=["age", "gender", "education", "apoe", 'boston', 'mmse'])
        # parser.add_argument('--tabular_continues_idx', nargs='+', type=int,
        #                     default=[0, 2, 4, 5])
        # parser.add_argument('--tabular_categorical_idx', nargs='+', type=int,
        #                     default=[1, 3])

        # - demographic information
        # parser.add_argument('--tabular', nargs='+', type=str,
        #                     default=['trailA', 'trailB', 'boston', 'animal',
        #                              'gds', 'mmse', 'npiq_DEL',	'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX',
        #                              'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE',
        #                              'npiq_APP', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
        #                              'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL',
        #                              'his_CVHATT', 'his_CBSTROKE', 'his_HYPERTEN', 'his_PSYCDIS', 'his_ALCOHOL'])
        # parser.add_argument('--tabular_continues_idx', nargs='+', type=int,
        #                     default=[0, 1, 3])
        # parser.add_argument('--tabular_categorical_idx', nargs='+', type=int,
        #                     default=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        #                              23, 24, 25, 26, 27, 28, 29, 30, 31, 32])

        # # - cognitive test scores
        # parser.add_argument('--tabular', nargs='+', type=str,
        #                     default=["age", "gender", "education", "apoe",
        #                              'his_CVHATT', 'his_CBSTROKE', 'his_HYPERTEN', 'his_PSYCDIS', 'his_ALCOHOL'])
        # parser.add_argument('--tabular_continues_idx', nargs='+', type=int,
        #                     default=[0, 2,])
        # parser.add_argument('--tabular_categorical_idx', nargs='+', type=int,
        #                     default=[1, 3, 4, 5, 6, 7, 8])

        # # - medical history
        # parser.add_argument('--tabular', nargs='+', type=str,
        #                     default=["age", "gender", "education", "apoe", 'trailA', 'trailB', 'boston', 'animal',
        #                              'gds', 'mmse', 'npiq_DEL',	'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX',
        #                              'npiq_ELAT', 'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE',
        #                              'npiq_APP', 'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
        #                              'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL'])
        # parser.add_argument('--tabular_continues_idx', nargs='+', type=int,
        #                     default=[0, 2, 3, 4, 5, 7])
        # parser.add_argument('--tabular_categorical_idx', nargs='+', type=int,
        #                     default=[1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        #                              27, 28, 29, 30, 31])

        # training parameters
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--epochs', type=int, default=40)
        parser.add_argument('--weight_decay', type=float, default=1e-8)

        # model parameters
        parser.add_argument('--dim', type=int, default=64)
        parser.add_argument('--trans_enc_depth', type=int, default=3)
        parser.add_argument('--heads_num', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)

        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        print(message)

        # save to the disk
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        expr_dir = os.path.join(opt.checkpoints_dir, f'{opt.task}_{opt.model}_{current_time}')
        mkdirs(expr_dir)

        opt.expr_dir = expr_dir

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        print(f'Create opt file opt.txt')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        self.parser = self.initialize(self.parser)
        self.opt = self.parser.parse_args()
        self.print_options(self.opt)
        return self.opt
