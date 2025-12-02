from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def drop_all_tables():
    try:
        with engine.connect() as conn:
            # Get all table names
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
            tables = [row[0] for row in result.fetchall()]
            
            if not tables:
                print("No tables found to delete")
                return
            
            print(f"Found {len(tables)} tables: {tables}")
            
            # Drop each table
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
                print(f"✅ Dropped table: {table}")
            
            conn.commit()
            print("🗑️ All tables deleted successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    drop_all_tables()