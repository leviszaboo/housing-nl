# Effect of Train Traffic and Train Stations on House Prices in the Netherlands

## Description
This project analyzes how the presence of train stations in municipalities and the volume of trains stopping at these stations affect house prices in the Netherlands. This study is part of a bachelor's thesis in Economics at the University of Amsterdam. 

The study consists of two main phases:
1. Analysis of the effect of the presence of a train station in a municipality on house prices per square meter, controlling for other factors. 
2. Analysis of the effect of the volume of train traffic (trains stopped at any station) in municipalities with a train station present on square meter house prices, controlling for other factors.

Click [here](data/output/main.csv) to view the main dataset.

## Data Sources
- **House Prices Data and Controls**: Regional indicators on the municipality level, located in the [data/unprocessed/cbs](data/unprocessed/cbs/) folder.
  - Source: [CBS Statline](https://opendata.cbs.nl/statline/#/CBS/nl/)
- **Municipalities GeoJSON**: Cartographic data in GeoJSON format on the outline of Dutch municipalities for 2023.
  - Source: [Cartomap](https://github.com/cartomap/nl)
- **Netherlands Train Stations**: Data on locations and categories of train stations across the Netherlands.
  - Source: [Rijden de Treinen (station data)](https://www.rijdendetreinen.nl/open-data/treinstations)
- **Netherlands Train Traffic**: Monthly data on train service across the Netherlands.
  - Source: [Rijden de Treinen (traffic data)](https://www.rijdendetreinen.nl/open-data/treinarchief)

## Project Stucture and File Descriptions
- `main.py`: Main entrypoint of the application.
- `src/dataset`: Directory containing files related to processing the raw data.
- `src/analysis`: Directory containing files related to the analysis of the [main dataset](data/output/main.csv). Contains two subfolders `phase_1` and `phase_2`
- `src/dataset/main.py`: Script that constructs the main dataset from the raw CBS data and the processed station and traffic data.
- `data/unprocessed`: Directory containing raw data files.
- `data/output`: Directory containing processed CSV files.
- `output`: Directory containing tables and figures resulting from the analysis.

## How to Run This Project

### Prerequisites
1. **Python**: Ensure that Python 3.x is installed. You can download it [here](https://www.python.org/downloads/).
2. **Virtual Environment**: It is recommended to use a virtual environment to manage dependencies. You can use `venv`, `virtualenv`, or `conda`.
3. **Dependencies**: The required Python libraries are listed in the `requirements.txt` file.

### Installation Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/leviszaboo/thesis.git
   ```
2. **Create and Activate Virtual Environment**:
   ```
   # Using venv
   python -m venv venv
   source venv/bin/activate  # For Unix/macOS
   venv\Scripts\activate.bat  # For Windows

   # Using conda (optional)
   conda create -n train-analysis python=3.x
   conda activate train-analysis
   ```
3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Run the Main Script**:
   * Execute the `main.py` script located at the top level of the project directory. This script will sequentially execute all necessary steps to prepare the dataset and run the analysis.

   ```
   python main.py
   ```
   * You can also specify the `--analysis_only`, `--dataset_only` or `--skip_station_data` command line arguments to run only a specific part of the main function. 
5. **Verify the Output**:
   * Check the generated files in the `data/output` folder for the main and stations dataset, and the `output` folder for the analysis results.

## License
"Effect of Train Traffic and Train Stations on House Prices in the Netherlands" © 2024 by Levente Szabo is licensed under [Creative Commons Attribution 4.0 International License][cc-by]. This means you are free to:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.
- **Conditions**: Attribution must be provided with proper credit, a link to the license, and indication if changes were made.

[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg



