from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    src_lang: str = "hi"
    tgt_lang: str = "en"


