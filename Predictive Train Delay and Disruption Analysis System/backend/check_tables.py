from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def check_tables():
    try:
        with engine.connect() as conn:
            # List all tables
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
            tables = result.fetchall()
            
            print("📊 Available tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            # Check each table data
            for table in tables:
                table_name = table[0]
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                count = count_result.fetchone()[0]
                print(f"\n📋 {table_name}: {count} records")
                
                if count > 0:
                    data_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3;"))
                    rows = data_result.fetchall()
                    for row in rows:
                        print(f"  {row}")
                        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_tables()