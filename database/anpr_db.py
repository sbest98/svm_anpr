import sqlite3
import datetime

def update_database(db, number_plate, car_park_id):
    # Open ANPR database
    db_conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    c = db_conn.cursor()
    # Insert new entry for ANPR call
    timestamp = datetime.datetime.now();
    c.execute('INSERT INTO anpr_live_feed (timestamp, car_number_plate_id, carpark_id) VALUES (?,?,?)',
              (timestamp, number_plate, car_park_id))
    # Commit and close
    db_conn.commit()
    c.close()
    db_conn.close()
