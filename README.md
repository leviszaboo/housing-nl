# Effect of Train Traffic and Train Stations on House Prices in the Netherlands

## Description
This project analyzes how the presence of train stations in municipalities and the volume of trains stopping at these stations affect house prices in the Netherlands. This study is part of a bachelor's thesis in Economics at the University of Amsterdam.

Click [here](data/output/main.csv) to view the main dataset.

## Data Sources
- **House Prices Data and Controls**: Regional indicators on the municipality level.
  - Source: [CBS Statline](https://opendata.cbs.nl/statline/#/CBS/nl/)
- **Netherlands Train Stations**: Data on locations and categories of train stations across the Netherlands.
  - Source: [Rijden de Treinen (station data)](https://www.rijdendetreinen.nl/open-data/treinstations)
- **Netherlands Train Traffic**: Monthly data on train service across the Netherlands.
  - Source: [Rijden de Treinen (traffic data)](https://www.rijdendetreinen.nl/open-data/treinstations)

## File Descriptions
- `src/dataset.py`: Script that constructs the main dataset from the raw data files.
- `data/unprocessed`: Directory containing raw data files.
- `data/output`: Directory containing processed CSV files.

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



