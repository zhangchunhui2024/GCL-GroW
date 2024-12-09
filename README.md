


## Dependencies

- python 3.9.15
- torch  2.3.1+cu118
- dgl    2.3.0+cu118
- ogb    1.3.6

## The main experiments

```
python main.py --dataname cora --epochs 100 --dfr 0.1 --der 0.6 --lr1 5e-4 --lr2 1e-2 --wd2 1e-4 --num_groups 32 --use_norm True
python main.py --dataname citeseer --dfr 0.0 --der 0.7 --lr2 0.05 --wd2 0.005 --num_groups 32 --epoch 50 --n_layers 1
python main.py --dataname pubmed --epochs 200 --dfr 0.2 --der 0.4 --lr2 1e-2 --wd2 1e-4 --num_groups 32
python main.py --dataname comp --num_groups 32 --epoch 50 --dfr 0.0 --der 0.7 --lr2 1e-2 --wd2 5e-4 --use_norm True
python main.py --dataname photo --num_groups 16 --epoch 50 --dfr 0.3 --der 0.7 --lr2 1e-2 --wd2 1e-3
```
## Multimodal Dataset experiments
- Please refer muti-dataset.py    muti-main.py
