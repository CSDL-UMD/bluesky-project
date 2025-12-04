from flask import Flask, jsonify, request
import psycopg2, re
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()
app = Flask(__name__)

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'port': int(os.getenv('DB_PORT', 5432)),
}
def fetch_data_from_database():
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query = "SELECT Day, TotalMessages, TotalLinks, NewsGreaterThan60, NewsLessThan60 FROM bsky_news"
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()
        return results

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        return {'error': str(e)}

async def fetch_urls_from_database(type_of_url, interval, score_threshold_min, score_threshold_max):
    try:
        valid_intervals = ['week', 'month']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}")
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        if interval == 'week':
            start_date = datetime.now() - timedelta(weeks=1)
        elif interval == 'month':
            start_date = datetime.now() - relativedelta(months=1)

        query = f"""
            SELECT {type_of_url}, SUM(count) AS total_count
            FROM newsguard_counts
            WHERE timestamp >= %s
            AND score >= %s AND score <= %s
            GROUP BY {type_of_url}
            ORDER BY total_count DESC
            LIMIT 10;
        """

        cursor.execute(query, (start_date, score_threshold_min, score_threshold_max))
        url_results = cursor.fetchall()
        cursor.close()
        conn.close()

        return url_results

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        raise

@app.route('/get_links', methods=['GET'])
async def get_links_data():
    try:
        type_of_url = request.args.get('type_of_url', default='url', type=str)
        interval = request.args.get('interval', default='day', type=str)
        score_threshold_min = request.args.get('score_threshold_min', default='60', type=int)
        score_threshold_max = request.args.get('score_threshold_max', default='100', type=int)

        data = await fetch_urls_from_database(type_of_url, interval, score_threshold_min, score_threshold_max)
        return jsonify(data)

    except Exception as e:
        app.logger.error(f"Error in /get_links: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_data', methods=['GET'])
def get_general_data():  # Renamed from get_data
    try:
        data = fetch_data_from_database()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.logger.info("Backend started")
    app.run(host='0.0.0.0', port=3001)
