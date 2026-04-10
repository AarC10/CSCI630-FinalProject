python3 code/main.py train \
  --input-dir ../dataset \
  --model minirocket \
  --model-kwargs estimator_name=ridge num_kernels=2000 max_dilations_per_kernel=32 class_weight=balanced \
  --sequence-length 100 \
  --stride 50 \
  --skip-learning-curves
