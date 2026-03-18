python3 code/main.py train \
  --input-dir ../dataset \
  --model knn \
  --model-kwargs num_kernels=1000 class_weight=balanced \
  --sequence-length 100 \
  --stride 50
