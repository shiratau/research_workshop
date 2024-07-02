# Project name

## Project overview
This repo contains three standalons RNN models that aim to learn and predict different problems, but use the same infrastructure and logic.
The models development order were *single* -> *multi* -> *feature*, since each model helped in understand the more advanced problem we tried to solve. 
Also, each model was desing after taking in considartion of previous model performance and prediction result.

 **Single** 
* The model is designed to predict σ (sd) from a sample taken from a normal distribution representing the subject's reaction times in an experiment.

**Multi**
* The model is designed to predict 3 σ (sd) of 3 different normal distributions (each representing reaction times of the same subject in the same experiment in chronological sequence under the assumption that they are different) from a series of numbers that are essentially a chain of the three samples corresponding to each distribution out of the three.

**Feature**
* The model is designed to improve the performance of the multi model which failed to solve the problem satisfactorily, in that in addition to the data described above that the multi model receives, this model also receives for each reaction time a timestamp that represents the time when the subject reacted (this is different from the reaction time itself).
This model is a 3D model so it supports adding features to data and in this case the feature we added is a timestamp.

> To understand more about how this project was developed, and the problem it tried to solve, checkout project_explanation.pptx


## Installation

### Prerequisites

- Python 3.x
- [Virtualenv](https://virtualenv.pypa.io/en/stable/installation/)

### Setup

#### Command Line

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
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

2. Enter the URL of your repository: `https://github.com/yourusername/your-repo-name.git` and click "Clone".

3. Once the project is opened, PyCharm may prompt you to create a virtual environment. If not, you can set it up manually:
   - Go to `File -> Settings -> Project: your-repo-name -> Python Interpreter`.
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
     python -m single.create_dataset [dataset_name]
     ```
   - For `multi`:
     ```bash
     python -m multi.create_dataset [dataset_name]
     ```
   - For `feature`:
     ```bash
     python -m feature.create_dataset [dataset_name]
     ```

   The `dataset_name` argument is optional.

#### PyCharm

1. Ensure the virtual environment is selected as the Python interpreter:
   - Go to `File -> Settings -> Project: your-repo-name -> Python Interpreter` and make sure the virtual environment is selected.

2. Open the dataset creation script for the desired sub-project in the editor:
   - `single/create_dataset.py`
   - `multi/create_dataset.py`
   - `feature/create_dataset.py`

3. Click on the green arrow next to the script's main function to run it.

### Running Tests

Each sub-project has a representative test file located under the `tests` directory. The tests verify that the model of the sub-project can run on the created dataset.

#### Command Line

1. Ensure the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Run the test for the desired sub-project using `pytest`:
   - For `single`:
     ```bash
     pytest tests/test_single.py::TestSingle::test_predict_from_dataset
     ```
   - For `multi`:
     ```bash
     pytest tests/test_multi.py::TestMulti::test_predict_from_dataset
     ```
   - For `feature`:
     ```bash
     pytest tests/test_feature.py::TestFeature::test_predict_from_dataset
     ```

#### PyCharm

1. Ensure the virtual environment is selected as the Python interpreter:
   - Go to `File -> Settings -> Project: your-repo-name -> Python Interpreter` and make sure the virtual environment is selected.

2. Open the test file for the desired sub-project in the editor:
   - `tests/test_single.py`
   - `tests/test_multi.py`
   - `tests/test_feature.py`

3. Click on the green arrow next to the `test_predict_from_dataset` method to run the test.
