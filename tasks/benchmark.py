from utilities.language_models import complete_text, complete_chat
from utilities.language_models import Chat, Chat_Message
import time
import logging

logging.basicConfig(level=logging.INFO)

start_stamp = time.time()
text = complete_text("Who's the greatest scientist of all time?")
end_stamp = time.time()
logging.info(f"Time elapsed: {end_stamp - start_stamp}")
logging.info(text)


chat = Chat(messages=[
    Chat_Message(role="system", content="You're a historian."),
    Chat_Message(role="user", content="Who is the greatest mathematician?"),
])

start_stamp = time.time()
text = complete_chat(chat)
end_stamp = time.time()
logging.info(f"Time elapsed: {end_stamp - start_stamp}")
logging.info(text)