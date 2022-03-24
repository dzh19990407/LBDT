BACKBONE=resnet50
TASK=a2d
MODEL_NAME=LBDT_4
NOW=$(date +"%Y%m%d_%H%M%S")
SAVE_PATH=./checkpoints/${TASK}/${MODEL_NAME}/${NOW}
if [ ! -d ${SAVE_PATH} ]; then
  mkdir -p ${SAVE_PATH}
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py \
--backbone ${BACKBONE} --model_name ${MODEL_NAME} --save_path ${SAVE_PATH} --task ${TASK} --interval 6 \
--dist \
2>&1 | tee ${SAVE_PATH}/log.txt
