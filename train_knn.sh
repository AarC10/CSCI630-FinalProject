python3 code/main.py train \
  --input-dir ../dataset \
  --model knn \
  --model-kwargs metric=dtw_metric\
  --sequence-length 100 \
  --stride 50
