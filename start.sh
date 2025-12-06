#!/bin/bash

uvicorn server:app --host 0.0.0.0 --port 8000 &
sleep 10
python ingest_to_mongodb.py
