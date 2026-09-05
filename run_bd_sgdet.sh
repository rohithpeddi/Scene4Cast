#!/bin/bash
PY=~/anaconda3/envs/scene4cast/bin/python
cd /home/rxp190007/CODE/Scene4Cast
for f in results/bucket_dumps/*sgdet*.pkl; do
  base=$(basename "$f" .pkl)
  $PY tools/bucketed_breakdown.py --dumps "$f" --out results/bd_parts/$base.json > results/bd_parts/$base.log 2>&1 &
done
wait
echo "SGDET PARTS DONE $(date +%H:%M:%S)"
