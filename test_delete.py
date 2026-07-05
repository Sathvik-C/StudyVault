from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)

try:
    with engine.begin() as conn:
        file_id = conn.execute(text("SELECT id FROM files LIMIT 1")).scalar()
        if not file_id:
            print("No file to delete.")
            exit(0)
            
        print(f"Deleting file {file_id}...")
        conn.execute(text("DELETE FROM document_chunks WHERE file_id = :fid"), {"fid": file_id})
        conn.execute(text("DELETE FROM important_messages WHERE file_id = :fid"), {"fid": file_id})
        conn.execute(text("DELETE FROM attachments WHERE file_id = :fid"), {"fid": file_id})
        conn.execute(text("DELETE FROM subjects WHERE file_id = :fid"), {"fid": file_id})
        conn.execute(text("DELETE FROM messages WHERE file_id = :fid"), {"fid": file_id})
        
        chat_id = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).scalar()
        conn.execute(text("DELETE FROM files WHERE id = :fid"), {"fid": file_id})

        if chat_id:
            remaining = conn.execute(text("SELECT COUNT(*) FROM files WHERE chat_id = :cid"), {"cid": chat_id}).scalar()
            if remaining == 0:
                conn.execute(text("DELETE FROM chats WHERE id = :cid"), {"cid": chat_id})
            else:
                conn.execute(text("UPDATE chats SET total_files = total_files - 1 WHERE id = :cid"), {"cid": chat_id})
        
        print("Delete SUCCESS.")
except Exception as e:
    print(f"FAILED: {e}")
