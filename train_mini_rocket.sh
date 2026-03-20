python3 code/main.py train \
  --input-dir ../dataset \
  --model minirocket \
  --model-kwargs num_kernels=1000 max_iter=10000 class_weight=balanced \
  --sequence-length 100 \
  --stride 50
