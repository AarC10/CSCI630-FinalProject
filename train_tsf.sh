python3 code/main.py train \
  --input-dir ../dataset \
  --model tsf \
  --model-kwargs class_weight=balanced max_iter=2000 \
  --sequence-length 100 \
  --stride 50 \
  --curve-fractions 0.1 0.2