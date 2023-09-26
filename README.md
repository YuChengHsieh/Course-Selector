# Course-Selector
Given all of the courses you want to choose several csv files with the columns: `Course No.`, `Course Title`, `Credit`, `Time`, `Average GPA`, `Compulsory or Optional`. See example in example csv file above. Our model will train a course selector with the Genetic Algorithm to help you choose courses optimally. The example csv table is listed below:
| Course No. | Course Title | Credit | Time   | Average GPA | Require / Option |
| ---------- | ------------ | ------ | ------ | ----------- | ---------------- |
| ESS100200  | 工程與系統科學探索    | 1      | W7W8   | 4           | 1                |
| ESS103001  | 工程力學         | 3      | M5M6R5 | 3.88        | 1                |
| ESS105000  | 能源與環境概論      | 2      | W3W4   | 4.04        | 0                |
## Getting Started
1. Prepare all of the courses you want to choose in several csv files.
1. clone this project
   ```bash
    git clone https://github.com/shieh322/Course-Selector
    ```
1. Create a virtual environment with `conda`
    ```bash
    conda create -n CS python=3.8
    ```
1. Activate the virtual environment
    ```bash 
    conda activate CS
    ```
1. Install required packages
    ```bash
    pip install -r requirements.txt
    ```
1. Run `AIcurriculum.py`
	```bash
	python AIcurriculum.py
	```
