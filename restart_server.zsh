# ps aux | grep uvicorn | awk '{print "kill -9 " $2}'
conda activate transformer
# uvicorn api:app --reload --host=0.0.0.0 --port=8082