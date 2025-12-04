<a name="readme-top"></a>
<br />
<div align="center">
<h3 align="center">MurkySky</h3>

  <p align="center">
    <br />
    <a href="https://csdl.umd.edu/murkysky/">View Demo</a>
  </p>
</div>

## About The Project

Bluesky, an innovative social media hub where users share posts, including news articles, is dedicated to upholding the integrity of information. This project monitors Bluesky posts, extracts URLs, and evaluates their credibility in real-time using NewsGuard's rating system. The findings are presented via a frontend application, providing users with valuable insights into the quality of news information on Bluesky.

### Built With

* [![Javascript][Javascript]][Javascript-url]
* [![Python][Python]][Python-url]
* [![PostgreSQL][PostgreSQL]][PostgreSQL-url]
* [![Shiny][Shiny]][Shiny-url]

## Getting Started

Below are the steps to set up your project locally and get it up and running smoothly.

### Prerequisites

This project requires [Node.js](http://nodejs.org/) and [npm](https://npmjs.org/). To make sure you have them available on your machine, try running the following command.
```sh
$ npm -v && node -v
```
Next, install [Python3](https://www.python.org/downloads/). On a Unix-based OS, the system's default Python installation is normally Python 2. To make sure you have it downloaded on your machine, try running the following command.
```sh
$ python3 --version
$ pip3 --version
```

If pip3 is not installed after installing Python3, then run the following command, and check again if it is installed.
```sh
$ python -m pip3 install --upgrade pip
$ pip3 --version
```

Finally, install Shiny for Python, and verify that it is downloaded on your machine.
```sh
$ pip3 install shiny
$ pip3 show shiny
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/CSDL-UMD/bluesky-project.git
   ```
2. Install npm packages
   ```sh
   npm install
   ```
3. Create a folder named "Messages" to store text files for posts, likes, and reposts. Additionally, create a folder named "NewsGuard" and include the following CSV files:
    * `all-sources-metadata.csv`
    * `metadata.csv` 
4. Enter your Bluesky and Database credentials in `.env`
   ```sh
    BLUESKY_USERNAME = 'Username'
    BLUESKY_PASSWORD = 'Password'
    DB_HOST = 'Remote Server'
    DB_USER = 'Database'
    DB_PASSWORD = 'Database Password'
   ```

### Deploy

## Firehose

To start collecting data from the firehose, run the following command:
   ```sh
   pm2 start index.js --name "bluesky"
   ```
Check that it is running successfully through the following command:
   ```sh
   pm2 list
   ```
## Frontend Application
To view the frontend application, run the following command.
   ```sh
   python3 -m uvicorn app:app --host 0.0.0.0 --port 3000 --reload
   ```
Access the application by entering http://127.0.0.1:3000/ in your web browser.


<!-- MARKDOWN LINKS & IMAGES -->
[Javascript]: https://shields.io/badge/JavaScript-F7DF1E?logo=JavaScript&logoColor=000&style=flat-square
[Javascript-url]: https://www.javascript.com/
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[PostgreSQL]: https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white
[PostgreSQL-url]: https://www.postgresql.org/
[Shiny]: https://img.shields.io/badge/Shiny-shinyapps.io-blue?style=flat&labelColor=white&&logoColor=blue
[Shiny-url]: https://shiny.posit.co/
