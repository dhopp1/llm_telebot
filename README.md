# llm_telebot
Create a Telegram chatbot to interact with a completely open-source local RAG or non-RAG LLM. Leverages the [local\_rag\_llm](https://github.com/dhopp1/local_rag_llm) library. Works well in conjunction with the [nlp_pipeline](https://github.com/dhopp1/nlp_pipeline) library for organizing and collecting your documents for RAG.

# Setup
1. clone this repo
2. get a [Hugging Face](https://huggingface.co/docs/api-inference/en/quicktour) API token and set it to the environment variable `HF_TOKEN`. [Windows](https://phoenixnap.com/kb/windows-set-environment-variable) and [Mac](https://phoenixnap.com/kb/set-environment-variable-mac) instructions.
3. create a [Telegram](https://telegram.org/) account
4. search for the "BotFather" bot in telegram
5. once there, send `/newbot` in the chat and follow the instructions, record your new bot's API token
6. `pip install -r requirements.txt`. If you want to use RAG, you will need to install PostgreSQL and pgvector per the instructions in the [local\_rag\_llm](https://github.com/dhopp1/local_rag_llm) library
7. if you're using RAG, create a `metadata.csv` file and gather your text files in a directory. The metadata csv file should have at least `text_id` (any unique ID) and a `file_path` (absolute file path of the .txt or .CSV file) column. Populate the `metadata/corpora_list.csv` file appropriately with the file paths.
8. if you want to use different base models, download them and update the `metadata/llm_list.csv` file accordingly. `llama-2-7b` is used by default.
9. if you want to use different DB credentials for your vector Postgres DB, you can edit them in the `metadata/db_creds.csv` file, or leave them at the default values.
10. add allowed users' Telegram IDs to `metadata/admin_list.csv`. [Instructions](https://bigone.zendesk.com/hc/en-us/articles/360008014894-How-to-get-the-Telegram-user-ID) on how to get Telegram user ID.
11. run the bot from the command line with `python bot.py <bot API key>`. The Bot API key is from step 5.
12. go to Telegram, search for your bot and start chatting. The default configuration will be with no RAG/contextualization.
13. Change base model and context corpus by sending directly in the chat e.g., `[reinitialize]{'new_llm':'mistral-7b', 'new_corpus':'corpus_name'}`. Where `'new_llm'` corresponds to a value in the `name` column of the `metadata/llm_list.csv` file, and `new_corpus` corresponds to a value in the `name` column of the `metadata/corpora_list.csv` file. Pass e.g, `[reinitialize]{'new_llm':'mistral-7b', 'new_corpus':None}` to again get a new bot with no RAG.
14. If you are using a RAG model, pass `cite your sources` within the text of your query to receive top N vector DB query results alongside the response. E.g., `Is this my question?`, will get you only a response, while `Is this my question? cite your sources`, will get the response pluse the vector DB query results.
15. If you want to tweak the model yourself, change the `initialize()` function in the `bot.py` file, according to the documentation in the [local\_rag\_llm](https://github.com/dhopp1/local_rag_llm) library.

