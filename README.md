# Brain-Computer Interface (BCI)

BCI is a Python-based project that aims to provide a platform for recording and processing EEG signals in real-time to develop applications for human-computer interaction using machine learning techniques. This project uses Emotiv EPOC+ headset and CyKIT library to acquire EEG signals.

The project is divided into two main components:
- Data Acquisition: This component handles the acquisition of raw EEG data from the Emotiv EPOC+ headset, and stores it in a CSV format.
- Data Processing: This component takes the raw EEG data acquired by the Data Acquisition component, performs preprocessing, feature extraction, and classification using machine learning techniques, and provides an output for interaction with various applications.

This README file is specific to the Data Acquisition component of the project.

## Data Acquisition

The Data Acquisition component is responsible for acquiring raw EEG signals from the Emotiv EPOC+ headset and storing it in CSV format. The data acquisition process involves flashing the LED light at a particular frequency to elicit SSVEP signals in the brain. The flickering light is used to generate an event-related potential that can be measured by the Emotiv EPOC+ headset.

The Data Acquisition component is built using Python and the following libraries:

- CyKIT (https://github.com/CymatiCorp/CyKit)

The main.py file in the data_acquisition directory of the project is the entry point for the Data Acquisition component. It contains the code to start the acquisition process and store the EEG data in CSV format. The config/config.py file contains the configuration parameters for the acquisition process, such as the stimulus duration, break duration, and frequencies.

## Project Structure

The project is organized into the following directories:

```
brain-computer-interface
├── data_acquisition
│   ├── config
│   │   └── config.py
│   ├── gui
│   │   ├── __init__.py
│   │   ├── MainWindow.py
│   │   ├── widgets
│   │   │   ├── FlickringModeGroupBox.py
│   │   │   ├── LogoLabel.py
│   │   │   └── __init__.py
│   │   └── resources
│   │       └── icon.png
│   ├── modules
│   │   ├── __init__.py
│   │   ├── emotiv.py
│   │   └── ssvep.py
│   ├── test
│   │   ├── __init__.py
│   │   ├── test_emotiv.py
│   │   └── test_ssvep.py
│   └── __init__.py
├── data_processing
│   ├── data
│   │   └── sample.csv
│   ├── model
│   │   ├── __init__.py
│   │   └── classifier.pkl
│   ├── __init__.py
│   ├── process_data.py
│   └── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

- data_acquisition: Contains the Data Acquisition component.
- data_processing: Contains the Data Processing component.
- datasets: Contains sample datasets for testing purposes.
- models: Contains machine learning models used for classification.
- scripts: Contains various scripts for data processing and analysis.

