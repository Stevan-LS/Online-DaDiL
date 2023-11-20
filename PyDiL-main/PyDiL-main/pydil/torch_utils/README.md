# Torch Utilities Module

This module contains utility functions related to Pytorch. Here you may find,

- ```data_utils.py```: utilities related to datasets
- ```architecture_utils.py```: definitions of neural network architectures
- ```supervised_learning.py```: Pytorch Lightning models for supervised learning
- ```unsupervised_learning.py```: Pytorch Lightning models for unsupervised learning


## Implemented Datasets

In ```vision_datasets.ObjectRecognitionDataset``` we implement the following datasets,

- [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/) [1]
- [Office Home](https://www.hemanthdv.org/officeHomeDataset.html) [2]
- [Adaptiope](https://gitlab.com/tringwald/adaptiope) [3]
- [DomainNet](http://ai.bu.edu/M3SDA/) [4]

In ```tabular_datasets.FeaturesDataset``` we implement datasets where samples are feature vectors in Rd.

## Coverage

In principle, ```ObjectRecognitionDataset``` can handle any dataset structured as follows,

```
dataset_name
├── domain_name
│   ├── class_name
│   │   ├── sample
├── folds
│   ├── domain_name_train_filenames.txt
│   ├── domain_name_test_filenames.txt
└──
```

Where sample refers to an image in a format readable by ```PIL.Image.open```

## References

[1] Saenko, K., Kulis, B., Fritz, M., & Darrell, T. (2010). Adapting visual category models to new domains. In Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV 11 (pp. 213-226). Springer Berlin Heidelberg.

[2] Venkateswara, H., Eusebio, J., Chakraborty, S., & Panchanathan, S. (2017). Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5018-5027).

[3] Ringwald, T., & Stiefelhagen, R. (2021). Adaptiope: A modern benchmark for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 101-110).

[4] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019). Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1406-1415).
