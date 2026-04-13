# Model Comparison

| Experiment | Model | Accuracy | Macro F1 | Weighted F1 | Train Time (s) | Inference Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| tsf | tsf | 0.9984 | 0.9824 | 0.9984 | 7632.6 | 2206.0 |
| minirocket_v3 | minirocket | 0.9884 | 0.8906 | 0.9896 | 9846.8 | 10.0 |
| minirocket_saga | minirocket | 0.9854 | 0.8125 | 0.9837 | 2323.8 | 8.9 |
| knn | knn | 0.9823 | 0.7532 | 0.9805 | 59.1 | 1860.8 |
| minirocket_v2 | minirocket | 0.9793 | 0.8366 | 0.9825 | 2186.2 | 9.9 |
| lr_lbgfgs_v2 | lr | 0.9522 | 0.6004 | 0.9468 | 454.3 | 0.7 |
| minirocket | minirocket | 0.9506 | 0.7334 | 0.9607 | 255.6 | 10.3 |
| lr_lbfgs | lr | 0.7486 | N/A | 0.8176 | 318.1 | 0.2 |
| catch22_lr | catch22_lr | 0.5426 | 0.2927 | 0.6566 | 980.1 | 131.7 |
