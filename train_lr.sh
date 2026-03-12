python3 code/main.py train \
  --input-dir ../dataset \
  --model lr \
  --model-kwargs class_weight=balanced solver=lbfgs max_iter=3000 \
  --sequence-length 100 \
  --stride 25