## Electricity load data (France)

The data is manually retrieved from the ENTSOE Open Data portal. The CSV file needs to be downloaded from an browser by clicking for the right fields in a form:

https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=12.10.2023+00:00|UTC|DAY&biddingZone.values=CTY|10YFR-RTE------C!BZN|10YFR-RTE------C&dateTime.timezone=UTC&dateTime.timezone_input=UTC#

- click the "Export Data" menu, then login / create and account;
- once logged-in, navigate back to the link above;
- make sure to select UTC time;
- tick the country of interest ("BZN|FR" in our case) in the left-hand panel;
- put a date within the year of interest (between 2020 and now);
- click the down arrow of the "Export Data" menu, and select "Total Load - Day
  Ahead / Actual (Year, CSV)".

## Weather data:

The `weather_<city>.parquet` files are the original Open-Meteo Historical
Forecast data (all variables, 2021-01-01 to 2025-05-31), fetched with the
`fetch_weather_data.py` script from the EuroSciPy 2025 example.

`weather_forecast_day1_<city>.parquet` and `weather_forecast_day2_<city>.parquet` contain
genuine 24h- and 48h-ahead temperature forecasts for the same cities and time
grid. They use the Open-Meteo Previous Runs API, GFS model, variables
`temperature_2m_previous_day1/2` (the value predicted 24h/48h before valid
time). JMA GSM fills the small GFS archive gaps (2021-01-01 to 2021-03-23 and
2023-12-30 to 2024-01-19). The pipeline defaults to
`fetch_city_weather_forecast(city, horizon)`, which uses day1 for horizons
<= 18h and day2 above that, and day2 when no horizon is given. Regenerate the files with:

```bash
python scripts/fetch_weather_forecasts.py
```

Compare the oracle and forecast sources with:

```bash
python scripts/inspection/plot_weather_forecasts.py
```

The comparison report is written to
`results/weather_forecast/oracle_vs_forecast_day1.html`.

Open-Meteo data is licensed under CC BY 4.0. The free API tier is for
non-commercial use; commercial use requires an Open-Meteo licence.