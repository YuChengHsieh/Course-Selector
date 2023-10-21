# Course-Selector

Given all of the courses you want to choose several CSV files with the columns: `Course No.`, `Course Title`, `Credit`, `Time`, `Average GPA`, `Compulsory or Optional`. See the example in the example CSV files above. Our model will train a course selector with the Genetic Algorithm to help you choose courses optimally. The example CSV table is listed below:
| Course No. | Course Title | Credit | Time | Average GPA | Require / Option |
| ---------- | ------------ | ------ | ------ | ----------- | ---------------- |
| ESS100200 | 工程與系統科學探索 | 1 | W7W8 | 4 | 1 |
| ESS103001 | 工程力學 | 3 | M5M6R5 | 3.88 | 1 |
| ESS105000 | 能源與環境概論 | 2 | W3W4 | 4.04 | 0 |

## Getting Started

1. Prepare all of the courses you want to choose in several CSV files.
2. Clone this project
   ```bash
    git clone https://github.com/shieh322/Course-Selector
   ```
3. Create a virtual environment with `conda`
   ```bash
   conda create -n venv python=3.8
   ```
4. Activate the virtual environment
   ```bash
   conda activate venv
   ```
5. Install required packages
   ```bash
   pip install -r requirements.txt
   ```
6. Run `main.py`
   ```bash
   python main.py
   ```

## Contributors

This project exists thanks to all people who contribute.

<a href="https://github.com/YuChengHsieh/Course-Selector/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=YuChengHsieh/Course-Selector" />
</a>
