MultiMolCGC.py: train a model without pretraining using MultiMolCGC model
usage:
```
python MultiMolCGC.py --data_csv [path to the dataset csv file for prediction] --output_path [csv path to save the prediction results.] 
```
example:
```
python MultiMolCGC.py --data_csv potency_train_val.csv --output_path cgc_pred.csv 
```

MultiMolCGC_pt.py: train a model with pretraining using MultiMolCGC model
usage:
```
python MultiMolCGC_pt.py --pretrain_csv [path to the CSV file used for pretraining.] --pretrain_size [pretraining size] --data_csv [path to the dataset csv file for prediction] --output_path [csv path to save the prediction results.] 
```

example:
```
python MultiMolCGC_pt.py --pretrain_csv vina_scores.csv --pretrain_size 50000 --data_csv potency_train_val.csv --output_path cgc_pred.csv 
```