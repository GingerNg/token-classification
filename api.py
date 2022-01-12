
from fastapi import FastAPI
from handlers import cck_ner


app = FastAPI()


@app.get("/")
def first_api():
    return "hello world"


@app.get("/cck_ner")
def ner(text: str = ""):
    try:
        res = cck_ner.infer(text)
        print(res)
        return {"res": res, 'status': 1}
    except Exception as e:
        print(e)
        return {"status": 0}



