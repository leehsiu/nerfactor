# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multiprocessing.spawn import get_preparation_data
from os.path import join, basename
from absl import app, flags
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil
from third_party.xiuminglib import xiuminglib as xm
from third_party.turbo_colormap import turbo_colormap_data, interpolate_or_clip


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_boolean('color_correct_albedo', False, "")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="eval")

def tonemapping(src,gamma=2.2):
    return src**(1./gamma)

def compute_rgb_scales(alpha_thres=0.9):
    """Computes RGB scales that match predicted albedo to ground truth,
    using just the first validation view.
    """
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # First validation view
    vali_dir = join(config_ini[:-4], 'vis_vali')
    data_root = config.get('DEFAULT', 'data_root')
    epoch_dirs = xm.os.sortglob(vali_dir, 'epoch?????????')
    epoch_dir = epoch_dirs[-1]
    batch_dirs = xm.os.sortglob(epoch_dir, 'batch?????????')
    batch_dir = batch_dirs[0]

    # Find GT path
    metadata_path = join(batch_dir, 'metadata.json')
    metadata = xm.io.json.load(metadata_path)
    view = metadata['id']
    pred_path = join(batch_dir, 'pred_albedo.png')
    gt_path = join(data_root, view, 'albedo.png')

    # Load prediction and GT
    pred = xm.io.img.read(pred_path) # gamma corrected
    gt = xm.io.img.read(gt_path) # linear
    pred = xm.img.normalize_uint(pred)
    gt = xm.img.normalize_uint(gt)
    pred = pred ** 2.2 # undo gamma
    gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')
    alpha = gt[:, :, 3]
    gt = gt[:, :, :3]

    # Compute color correction scales, in the linear space
    is_fg = alpha > alpha_thres
    opt_scale = []
    for i in range(3):
        x_hat = pred[:, :, i][is_fg]
        x = gt[:, :, i][is_fg]
        scale = x_hat.dot(x) / x_hat.dot(x_hat)
        opt_scale.append(scale)
    opt_scale = tf.convert_to_tensor(opt_scale, dtype=tf.float32)

    return opt_scale





#albedo
#normal
#rgba_{}
#pred_rgb_probes_{}

def cal_psnr(gt_path,pred_path,fg,gamma=False):
    # Load prediction and GT
    pred = xm.io.img.read(pred_path) # gamma corrected
    gt = xm.io.img.read(gt_path) # linear
    pred = xm.img.normalize_uint(pred)
    gt = xm.img.normalize_uint(gt)
    gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')
    gt = gt[...,:3]
    if gamma:
        gt = tonemapping(gt)

    gt_fg = gt[fg,:]
    pred_fg = pred[fg,:]
    mse = np.mean((gt_fg - pred_fg)**2)
    psnr = -10.*np.log(mse)/np.log(10.)
    return psnr

def cal_normal(gt_path,pred_path,fg):
    # Load prediction and GT
    pred = xm.io.img.read(pred_path) # gamma corrected
    gt = xm.io.img.read(gt_path) # linear
    pred = xm.img.normalize_uint(pred)
    gt = xm.img.normalize_uint(gt)
    gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')
    gt = gt[...,:3]


    gt_fg = gt[fg,:]*2.0 - 1.0
    pred_fg = pred[fg,:]*2.0 - 1.0

    gt_fg = xm.linalg.normalize(gt_fg,1)
    pred_fg = xm.linalg.normalize(pred_fg,1)

    dot = np.sum(gt_fg*pred_fg,axis=1)
    dot = np.clip(dot,-1.,1.0)
    dot_mean = np.mean(dot)

    return np.arccos(dot_mean)/np.pi*180






def eval(batch_dirs,alpha_thres=0.9):
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)
    data_root = config.get('DEFAULT', 'data_root')

    psnr_albedo = []
    psnr_relight = []
    psnr_fv = []
    err_normal = []


    import glob
    all_lights_file = glob.glob(join(batch_dirs[0],'*probes*'))
    all_lights = [basename(el)[16:-4] for el in all_lights_file]

    for batch_dir in batch_dirs:
        #Find GT path
        metadata_path = join(batch_dir, 'metadata.json')
        metadata = xm.io.json.load(metadata_path)
        view = metadata['id']


        pred_path = join(batch_dir, 'pred_albedo.png')
        gt_path = join(data_root, view, 'albedo.png')

        # Load prediction and GT
        gt = xm.io.img.read(gt_path) # linear
        pred = xm.io.img.read(pred_path)
        gt = xm.img.normalize_uint(gt)
        gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')
        alpha = gt[:, :, 3]
        fg = alpha > alpha_thres

        psnr = cal_psnr(gt_path,pred_path,fg,True)
        psnr_albedo.append(psnr)


        pred_path = join(batch_dir, 'pred_rgb.png')
        gt_path = join(data_root, view, 'rgba.png')
        
        psnr = cal_psnr(gt_path,pred_path,fg,False)
        psnr_fv.append(psnr)

        for light in all_lights:
            pred_path = join(batch_dir, f'pred_rgb_probes_{light}.png')
            gt_path = join(data_root, view, f'rgba_{light}.png')
            # Load prediction and GT
            psnr = cal_psnr(gt_path,pred_path,fg,False)
            psnr_relight.append(psnr)

        pred_path = join(batch_dir, 'pred_normal.png')
        gt_path = join(data_root, view, 'normal.png')
        err = cal_normal(gt_path,pred_path,fg)
        err_normal.append(err)


    print('albedo',np.mean(psnr_albedo))
    print('fv',np.mean(psnr_fv))
    print('relight',np.mean(psnr_relight))
    print('normal',np.mean(err_normal))




    # Compute color correction scales, in the linear space

    return



def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_eval', basename(FLAGS.ckpt))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'vali', debug=FLAGS.debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    ioutil.restore_model(model, FLAGS.ckpt)

    # Optionally, color-correct the albedo
    albedo_scales = None
    if FLAGS.color_correct_albedo:
        albedo_scales = compute_rgb_scales()

    #For all test views
    logger.info("Running inference")
    for batch_i, batch in enumerate(
            tqdm(datapipe, desc="Inferring Views", total=n_views)):
        # Inference
        _, _, _, to_vis = model.call(
            batch, mode='vali', relight_probes=True,
            albedo_scales=albedo_scales)
        # Visualize
        outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))
        model.vis_batch(to_vis, outdir, mode='vali')
        # Break if debugging
        if FLAGS.debug:
            break

    #calculate metrics
    batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
    eval(batch_vis_dirs)



if __name__ == '__main__':
    app.run(main)



