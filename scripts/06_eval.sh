#scene='hotdog_2163'
#scene='lego_3072'
#scene='drums_3072'
scene='ficus_2188'
gpus='0'
model='nerfactor'
proj_root='/home/v-xiuli1'
repo_dir="$proj_root/workspace/nerfactor"
viewer_prefix='' # or just use ''
data_root="$proj_root/databag/nerf/nerfactor/$scene"
outroot="$proj_root/weights/nerfactor/${scene}_$model"
ckpt="$outroot/lr5e-3/checkpoints/ckpt-10"

if [[ "$scene" == pinecone || "$scene" == vasedeck || "$scene" == scan* ]]; then
    # Real scenes: NeRF & DTU
    color_correct_albedo='false'
else
    color_correct_albedo='true'
fi
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/eval_run.sh" "$gpus" --ckpt="$ckpt" --color_correct_albedo="$color_correct_albedo"