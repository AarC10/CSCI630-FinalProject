python3 code/main.py train \
  --input-dir ../dataset \
  --model catch22_lr \
#  --max-files 2000 \
  --model-kwargs class_weight=balanced solver=lbfgs tol=1e-2 max_iter=3000 \
  --sequence-length 100 \
  --stride 25