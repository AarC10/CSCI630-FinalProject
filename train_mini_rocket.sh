python3 code/main.py train \
  --input-dir ../dataset \
  --model minirocket \
  --model-kwargs num_kernels=1000 solver=saga max_iter=3000 tol=1e-4 \
  --sequence-length 100 \
  --stride 50
