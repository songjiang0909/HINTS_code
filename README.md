HINTS
-----------------
This is the implementation of our paper "[HINTS: Citation Time Series Prediction for New Publications via Dynamic Heterogeneous Information Network Embedding](http://web.cs.ucla.edu/~yzsun/papers/2021_WWW_HINTS.pdf)".

Data
-----------------

The original data used could be access from [Aminer](https://www.aminer.org/citation)  [APS](https://journals.aps.org/datasets).

We also provide our processed data to reproduce the results reported in our paper. [preprocessed
data]()



Requirement
----------------------
All required packages could be found in `requirements.txt`.

NOTE: our implementation is on Tensorflow 1.14, may not be very friendly if you are familar with dynamic computational graph based frameworks.



How to run?
----------------------

* Step0 (data): 
	* Download the [processed data]() and `unzip *.zip` under the root folder.

* Step1 (run):
	* `cd src`
	* For Aminer dataset: `python main.py --dataset aminer --epochs 700 --batch_size 3000`
	* For APS dataset: `python main.py --dataset aps --epochs 500 --batch_size 1200`

Arguments interpretation:

- `--dataset`: processed dataset, either AMiner or APS. 

- `--epochs`: number of training on the training set.

- `--batch_size` : batch size of one training.


Note that the batch size in set as the number of papers used in training, i.e., there is only one interation per epoch. I didn't tune too much on the hypermeters.


The prediction files will be stored under the `result` folder.




Contact
----------------------
Song Jiang <songjiang@cs.ucla.edu>



How to run?
----------------------

```bibtex
@inproceedings{LG-ODE,
  title={HINTS: Citation Time Series Prediction for New Publications via Dynamic Heterogeneous Information Network Embedding},
  author={Song Jiang, Bernard J. Koch, Yizhou Sun},
  booktitle={Proceedings of The Web Conference},
  year={2021}
}
```


