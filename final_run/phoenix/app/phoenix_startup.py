# /app/phoenix_startup.py
import os, signal, time, phoenix as px

WORKDIR = os.getenv("PHOENIX_WORKING_DIR", "/app/.phoenix")
EXPORTS = os.getenv("PHOENIX_EXPORT_DIR", f"{WORKDIR}/exports")
os.makedirs(EXPORTS, exist_ok=True)

print("🚀 Starting local Phoenix…")
session = px.launch_app(
    host="0.0.0.0",
    port=6006,
    use_temp_dir=False,          # write phoenix.sqlite into WORKDIR
)

print(f"✅ Phoenix UI → http://localhost:6006")
print("   SQLite file  :", os.path.join(WORKDIR, 'phoenix.sqlite'))
print("   Export folder:", EXPORTS)

# graceful shutdown so DB is flushed and Parquet written
def _shutdown(*_):
    print("\n🛑  Stopping Phoenix…")
    try:
        print("   ↳ exporting TraceDataset to Parquet …")
        trace_ds = session.client.get_trace_dataset()
        trace_ds.save(directory=EXPORTS)        # parquet+metadata.json
    except Exception as e:
        print("   export failed:", e)

    session.end()
    exit(0)

signal.signal(signal.SIGTERM, _shutdown)   # docker stop / compose down
signal.signal(signal.SIGINT,  _shutdown)   # Ctrl-C

while True:
    time.sleep(1)

