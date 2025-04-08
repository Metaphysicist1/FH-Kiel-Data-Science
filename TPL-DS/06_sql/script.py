from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("MYSQL_USER")
PASSWORD = os.getenv('MYSQL_PASSWORD')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DATABASE = os.getenv('MYSQL_DATABASE')

# Debug prints
print(f"Username: {USERNAME}")
print(f"Host: {HOST}")
print(f"Port: {PORT}")
print(f"Database: {DATABASE}")

engine = create_engine(f'mysql+mysqlconnector://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?auth_plugin=mysql_native_password')

connection = engine.connect()

# print(connection)
# result = connection.execute(text("SELECT * FROM tracks"))

# print(result.fetchall())

inspector = inspect(engine)

print('Available Databases:')
print(inspector.get_schema_names())