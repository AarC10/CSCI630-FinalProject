python3 code/main.py train \
  --input-dir ../dataset \
  --model minirocket \
  --model-kwargs num_kernels=1000 solver=saga max_iter=3000 tol=1e-4 class_weight=balanced \
  --sequence-length 100 \
  --stride 50 \
  --label-strategy center \
  --split-strategy group \
  --smoothing-window 5 \
  --cv-folds 5
