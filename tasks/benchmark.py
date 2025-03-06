from utilities.language_models import llm_function
import time
import logging

logging.basicConfig(level=logging.INFO)

start_stamp = time.time()
text = llm_function("Who's the greatest scientist of all time?")
end_stamp = time.time()
logging.info(f"Time elapsed: {end_stamp - start_stamp}")
logging.info(text)
