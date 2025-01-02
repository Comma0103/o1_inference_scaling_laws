import time
from call_gemini import Gemini
import google.generativeai as genai

if __name__ == '__main__':

    for m in genai.list_models():
        print(m)
    print("\n\n\n")

    tic = time.time()
    gemini_model = Gemini(
        model_name="gemini-2.0-flash-thinking-exp-1219",
        sys_inst="You are a helpful and harmless assistant. You are Gemini developed by Google. You should think step-by-step."
    )
    res = gemini_model.generate(content="Explain how AI works", temperature=0.8, top_p=0.9, return_completion=True)
    toc = time.time()
    # print(res)
    print("Response text:", res.text)
    print("Response token count:", res.usage_metadata.candidates_token_count)
    print(f"Response time: {toc - tic:.2f} seconds")