from transformers import pipeline
from fastapi import FastAPI
from mangum import Mangum

from pydantic import BaseModel
from typing import List

class BodyModel(BaseModel):
    sentence: List[str]
      
import torch
model = torch.load('./ner_model_nerda_V2.pth')

app = FastAPI(title='Serverless Lambda FastAPI', root_path="/Prod/")

@app.post("/NER", tags=["NER"])
def NER( item: BodyModel):
    sentence = item.sentence
    return {'result': model.predict_text(sentence)}
