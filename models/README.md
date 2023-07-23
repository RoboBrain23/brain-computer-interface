## EEG Machine Learning Model

The EEG machine learning model is designed to analyze EEG (Electroencephalogram) data and make predictions based on the recorded brain activity. This repository provides implementation of five distinct models for EEG analysis. Researchers and developers can explore and choose the most suitable model based on their specific EEG analysis requirements.

### Implemented Models

The repository includes the following five distinct models for EEG analysis:

- [TCNN Model](https://ieeexplore.ieee.org/document/9632600).
- [FBTCNN Model](https://ieeexplore.ieee.org/document/9632600).
- [SSVEPNET Model](https://iopscience.iop.org/article/10.1088/1741-2552/ac8dc5/meta).
- [EEGNET Model](http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8).
- [DeepConvNet Model](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730).
- [ShallowConvNet Model](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730).

Researchers and developers can explore and choose the most suitable model based on their specific EEG analysis requirements. Each model is implemented to work with the specified data format and can provide accurate predictions and insightful analysis of EEG data.


### Data Format

The input data for the EEG model consists of three components:

1. Data File:
   - The data file should contain raw EEG data without any time separation.
   - The shape of the data file should be `(num_rows, num_channels)`.

2. Time Separation File:
   - The time separation file is a separate file that contains information about the time gaps between different trials.
   - The shape of the time separation file should be `(num_trials, 1)`.

3. Label File:
   - The label file should contain the corresponding labels for each trial in the EEG data.
   - The shape of the label file should match the time separation file, i.e., `(num_trials, 1)`.

### Getting Started

To use the EEG machine learning model and explore the implemented models you can see the [Example](example.ipynb).

### Contributions

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open a [GitHub Issue](https://github.com/RoboBrain23/brain-computer-interface/issues) or submit a pull request.

### License

This project is licensed under the [MIT License](./LICENSE). Feel free to use and modify the code for your needs while respecting the terms of the license.
