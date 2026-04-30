import os
import uvicorn

if __name__ == "__main__":
    port    = int(os.environ.get("PORT", 7860))           # 7860 = standard for Hugging Face Spaces
    workers = int(os.environ.get("WEB_CONCURRENCY", 1))   # Keep 1 — models are large singletons
    dev     = os.environ.get("APP_ENV", "production").lower() == "development"

    print(f"Starting InSightATS API  •  env={'dev' if dev else 'prod'}  •  port={port}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",          # Required: Nginx proxies from outside → 0.0.0.0
        port=port,
        reload=dev,              # reload only when APP_ENV=development
        workers=workers,
    )

