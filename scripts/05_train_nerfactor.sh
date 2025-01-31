#scene='lego_3072'
#scene='hotdog_2163'
#scene='drums_3072'
scene='ficus_2188'
gpus='0'
model='nerfactor'
overwrite='True'
proj_root='/home/v-xiuli1'
repo_dir="$proj_root/workspace/nerfactor"
viewer_prefix='' # or just use ''

# I. Shape Pre-Training
data_root="$proj_root/databag/nerf/nerfactor/$scene"
if [[ "$scene" == scan* ]]; then
    # DTU scenes
    imh='256'
else
    imh='512'
fi
if [[ "$scene" == pinecone || "$scene" == vasedeck || "$scene" == scan* ]]; then
    # Real scenes: NeRF & DTU
    near='0.1'; far='2'
else
    near='2'; far='6'
fi
if [[ "$scene" == pinecone || "$scene" == vasedeck || "$scene" == scan* ]]; then
    # Real scenes: NeRF & DTU
    use_nerf_alpha='True'
else
    use_nerf_alpha='False'
fi
surf_root="$data_root/geometry"
shape_outdir="$proj_root/weights/nerfactor/${scene}_shape"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config='shape.ini' --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,outroot=$shape_outdir,viewer_prefix=$viewer_prefix,overwrite=$overwrite"

# II. Joint Optimization (training and validation)
shape_ckpt="$shape_outdir/lr1e-2/checkpoints/ckpt-2"
brdf_ckpt="$proj_root/weights/brdf/lr1e-2/checkpoints/ckpt-50"
if [[ "$scene" == pinecone || "$scene" == vasedeck || "$scene" == scan* ]]; then
    # Real scenes: NeRF & DTU
    xyz_jitter_std=0.001
else
    xyz_jitter_std=0.01
fi
test_envmap_dir="$proj_root/databag/nerf/light-probes/test"
shape_mode='finetune'
outroot="$proj_root/weights/nerfactor/${scene}_$model"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,near=$near,far=$far,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,shape_model_ckpt=$shape_ckpt,brdf_model_ckpt=$brdf_ckpt,xyz_jitter_std=$xyz_jitter_std,test_envmap_dir=$test_envmap_dir,shape_mode=$shape_mode,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite"


