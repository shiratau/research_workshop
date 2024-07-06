# Research Workshop: RTV Change model

- [Project overview](#project-overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Dataset Creation](#dataset-creation)
  - [Running Tests](#running-tests)
  - [Test Results](#test-results)
- [Contributing](#contributing)
- [Contact](#contact)
- [Thanks](#thanks)

## Project overview
This repo contains three standalons RNN models that aim to learn and predict different problems, but use the same infrastructure and logic.
The models development order was *single -> multi -> feature*, since each model helped in understand the more advanced problem we tried to solve by each model. 
Also, each model was design after taking in consideration previous model performance and prediction result.

 **Single** 
* The model is designed to predict σ (sd) from a sample taken from a normal distribution representing the subject's reaction times in an experiment.

**Multi**
* The model is designed to predict 3 σ (sd) of 3 different normal distributions (each representing reaction times of the same subject in the same experiment in chronological sequence under the assumption that they are different) from a series of numbers that are essentially a chain of the three samples corresponding to each distribution out of the three.

**Feature**
* The model is designed to improve the performance of the *multi* model which currently failed to solve the problem satisfactorily. 
* In addition to the data described above that the *multi* model receives. This model also receives for each reaction time a timestamp that represents the time when the subject reacted (this is different from the reaction time itself).
* This model is a 3D model, so it supports adding features to data, and in this case the feature we added is a timestamp.

> **Note 1**: The correct distribution form to represent RT of a subject in an experiment is ex-gaussian distribution. We choose to start with this simplification of the problem in order to focus on model development, ideally this project will continue to grow with a model able to predict τ (tau) of ex-gaussian distribution.

> **Note 2**: To understand more about how this project was developed, and the problem it tried to solve, checkout [presentation.pptx](todo)


## Installation

### Prerequisites

- Python 3.11
- [Virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)

### Setup

#### Command Line

1. Clone the repository:
   ```bash
   git clone https://github.com/shiratau/research_workshop.git
   cd research_workshop
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On MacOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### PyCharm

1. Open PyCharm and select "Get from VCS".

2. Enter the URL of your repository: `https://github.com/shiratau/research_workshop.git` and click "Clone".

3. Once the project is opened, PyCharm may prompt you to create a virtual environment. If not, you can set it up manually:
   - Go to `File -> Settings -> Project: research_workshop -> Python Interpreter`.
   - Click on the gear icon and select `Add...`.
   - Choose `New environment` using Virtualenv and ensure the base interpreter is set to Python 3.x.
   - Click `OK`.

4. PyCharm should detect the `requirements.txt` file and prompt you to install the dependencies. If not, right-click on `requirements.txt` and select `Install`.

## Usage

The project is divided into three sub-projects: `single`, `multi`, and `feature`. Each sub-project has a dataset creation script and a corresponding test.

### Dataset Creation

#### Command Line

1. Ensure the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Run the dataset creation script for the desired sub-project:
   - For `single`:
     ```bash
     python -m single.create_datasets_single_sd [dataset_name]
     ```
   - For `multi`:
     ```bash
     python -m multi.create_datasets_multi_sd [dataset_name]
     ```
   - For `feature`:
     ```bash
     python -m feature.create_datasets_with_feature [dataset_name]
     ```

   The `dataset_name` argument is optional.

#### PyCharm

1. Ensure the virtual environment is selected as the Python interpreter:
   - Go to `File -> Settings -> Project: research_workshop -> Python Interpreter` and make sure the virtual environment is selected.

2. Open the dataset creation script for the desired sub-project in the editor:
   - `single/create_datasets_single_sd.py`
   - `multi/create_datasets_multi_sd.py`
   - `feature/create_datasets_with_feature.py`

3. Click on the green arrow next to this line in the file:
    ```python
    if __name__ == '__main__':
    ```

#### Configuration

Each dataset can be configured before creation through the const in the relevant creation file, for example:

```Python
NUMBER_OF_AGENTS = 100
NUMBER_OF_DIST_PER_AGENT = 3
MAX_SIZE_OF_SAMPLE = 210
MIN_SIZE_OF_SAMPLE = 190
MU_RANGE = (20.1, 20.8)
SIGMA_RANGE = (0.01, 0.1)
SHOULD_ROUND = True
N_DIGIT = 5
```

#### Output
Dataset will be saved as CSV file under `[sub-project_name]/datasets/`, and a metadata file of the set configuration will be saved as TXT file under `[sub-project_name]/datasets/metadata/`.


### Running Tests

* Each sub-project has a representative test file located under the `tests` directory. 
* The tests verify that the model of the sub-project can run on the created dataset.
* Each sub-project also have basic sanity tests in it test file.

#### Command Line

1. Ensure the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Run the test for the desired sub-project using `pytest`:
   - For `single`:
     ```bash
     pytest .\tests\test_predictions_single.py::test_predict_from_dataset --file_id [dataset_file_id]
     ```
   - For `multi`:
     ```bash
     pytest .\tests\test_predictions_multi.py::test_predict_from_dataset --file_id [dataset_file_id]
     ```
   - For `feature`:
     ```bash
     pytest .\tests\test_predictions_feature.py::test_predict_from_dataset --file_id [dataset_file_id]
     ```
The `--file_id` flag is optional, running test without it will run the test on the default dataset file configured in the file test: `DEFAULT_TEST_FILE_ID = [default_dataset]`.

#### PyCharm

1. Ensure the virtual environment is selected as the Python interpreter:
   - Go to `File -> Settings -> Project: research_workshop-> Python Interpreter` and make sure the virtual environment is selected.

2. Open the test file for the desired sub-project in the editor:
   - `tests/test_predictions_single.py`
   - `tests/test_predictions_multi.py`
   - `tests/test_predictions_feature.py`

3. Click on the green arrow next to the `test_predict_from_dataset` method to run the test.

**Note**: this will run the test on the dafault dataset file, to run on different dataset, change the `DEFAULT_TEST_FILE_ID` value.

### Test Results

Test results will be saved in CSV files under the `tests/results/` directory:

- For `single` sub-project: `tests/results/single/<dataset_name>_<timestamp>.csv`
- For `multi` sub-project: `tests/results/multi/<dataset_name>_<timestamp>.csv`
- For `feature` sub-project: `tests/results/feature/<dataset_name>_<timestamp>.csv`

Make sure to check the respective directory for the results of your tests.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b branch-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add these changes: <elaborate>'`).
5. Push to the branch (`git push origin branch-name`).
6. Open a pull request.

## Contact

E-mail: [shiraevron@mail.tau.ac.il](mailto:shiraevron@mail.tau.ac.il)

Project Link: [https://github.com/shiratau/research_workshop](https://github.com/shiratau/research_workshop)

## Thanks


- [Dr. Nitzan Shahar](https://github.com/NitzanShahar): The PI of the [Shahar Lab](https://www.shahar-lab.com/), for hosting this workshop project.
- [Gili Katabi](https://github.com/gilikatabi): PhD. Student, mentored this project.
- Eliah Golan: Co-workshop-researcher, conducted the literature review for this project and co-presented it at a lab meeting.


