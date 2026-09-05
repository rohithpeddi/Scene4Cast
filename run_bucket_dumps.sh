#!/bin/bash
set -e
PY=~/anaconda3/envs/scene4cast/bin/python
cd /home/rxp190007/CODE/Scene4Cast
CFGS="
configs/methods/predcls/w_sttran_predcls_resnet50.yaml
configs/methods/predcls/w_sttran_pp_predcls_resnet50.yaml
configs/methods/predcls/w_dsgdetr_predcls_resnet50.yaml
configs/methods/predcls/w_dsgdetr_pp_predcls_resnet50.yaml
configs/methods/predcls/worldwise_predcls_dinov3l.yaml
configs/methods/sgdet/w_sttran_sgdet_resnet50.yaml
configs/methods/sgdet/w_sttran_pp_sgdet_resnet50.yaml
configs/methods/sgdet/w_dsgdetr_sgdet_resnet50.yaml
configs/methods/sgdet/w_dsgdetr_pp_sgdet_resnet50.yaml
configs/methods/sgdet/worldwise_sgdet_dinov3l.yaml
"
for c in $CFGS; do
  echo "########## $(date +%H:%M:%S)  $c"
  CUDA_VISIBLE_DEVICES=0 $PY tools/dump_predictions.py --config "$c" --ckpt checkpoint_19 --frames all 2>&1 | grep -vE "FutureWarning|torch.load|weights_only|SECURITY|Dumping:" | tail -4
done
echo "########## ALL DUMPS DONE $(date +%H:%M:%S)"
