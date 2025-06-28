from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neural_router import ModelRouter
import uvicorn

app = FastAPI()
router = ModelRouter()

class RequestModel(BaseModel):
    prompt: str
    model_type: str = "mistral"

@app.post("/generate")
async def generate_text(request: RequestModel):
    try:
        if request.model_type != router.current_model:
            router.switch_model(request.model_type)
        
        response = router.generate_response(request.prompt)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
