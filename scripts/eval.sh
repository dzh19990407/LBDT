BACKBONE=resnet50
TASK=a2d
MODEL_NAME=LBDT_4
SAVE_PATH=./checkpoints/${TASK}/${MODEL_NAME}
CHECKPOINT=a2d_sota_layer4
RESUME_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/dingzihan/cvpr22/checkpoints/a2d/${CHECKPOINT}.ckpt
NOW=$(date +"%Y%m%d_%H%M%S")

if [ ! -d ${SAVE_PATH} ]; then
  mkdir -p ${SAVE_PATH}
fi

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 main.py \
--backbone ${BACKBONE} --model_name ${MODEL_NAME} --save_path ${SAVE_PATH} --resume $RESUME_PATH \
--task ${TASK} --interval 6  --batch_size 8 --dist
