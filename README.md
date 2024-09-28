<img width="181" alt="image" src="https://github.com/user-attachments/assets/6c47fb71-ac0a-4b30-83fe-7d9de922027b">


# Using machine learning methods to predict electric vehicle penetration in the automotive market
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#data-collectors">Data Collectors</a></li>
    <li><a href="#modeling">Modeling</a></li>
    <li><a href="#results-evaluation">Results Evaluation</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Electric vehicles (EVs) have been introduced as an alternative to gasoline and diesel cars to reduce greenhouse gas emissions, optimize fossil fuel use, and protect the environment. Forecasting the sale of EVs and their penetration into the automotive market has been a significant issue for governments, policymakers, and car manufacturers to plan the production of EVs, set proper policies, and provide sufficient energy and infrastructure. The main goal of my [article](https://www.nature.com/articles/s41598-023-35366-3) is to apply Machine Learning (ML) methods to build an efficient prediction model to estimate the sale of all vehicles in the dataset, the share of EVs in each segment, and determine the main factors that influence the sales of each EV. The codes related to the article are presented in three steps.

<!-- DATA COLLECTORS -->
## Data Collectors
The primary dataset contains monthly information about 357 vehicles, such as brand (or "make" in auto industry lingo, e.g., Benz), model, segmentation, category, shoppers, and sales of different types of cars in the United States from 2014 to 2020. Other information has been extracted based on the vehicles in this dataset:
- The economic_data_collector.py file extracts economic indicators such as Gross Domestic Product (GDP), Available Personal Income, Consumer Price Index, Interest Rate, Unemployment Rate, Industrial Investment Demand, Petroleum Charge, Private Consumption, and Latent Replacement Demand from the Federal Reserve Economic Data website.
- The google_trends_mm_collector.py file extracts Google trends data for a specified keyword ("Make" + "Model"). The keyword has been selected for Google trend data to evaluate the number of searches for each car from 2014 to 2020 and for the United States of America.
- The news_collector_wesite1.py file extracts daily news published from 2014 to 2020 on the [Automotive](https://www.autonews.com/) News website.
- The vehicle_specification_collector.py file extracts vehicle specification data from the [Thecarconnection](https://www.thecarconnection.com/) website.

<!-- DATA COLLECTORS -->
## Modeling
data

<!-- RESULTS EVALUATION -->
## Results Evaluation
results
