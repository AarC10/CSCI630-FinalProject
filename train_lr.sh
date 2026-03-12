python3 code/main.py train \
  --input-dir ../dataset \
  --model lr \
  --max-files 2000 \
  --model-kwargs class_weight=balanced solver=saga tol=1e-2 max_iter=500 \
  --sequence-length 100 \
  --stride 25