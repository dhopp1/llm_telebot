import pandas as pd
from local_rag_llm import local_llm
import telebot
import os
import sys
import gc

# bot setup
BOT_TOKEN = sys.argv[1]

bot = telebot.TeleBot(BOT_TOKEN)

# List of user_id of authorized users
admins = list(pd.read_csv("metadata/admin_list.csv").user_id)

# setting up LLM
# parameters/authentication
llm_dict = pd.read_csv("metadata/llm_list.csv")
corpora_dict = pd.read_csv("metadata/corpora_list.csv")
db_info = pd.read_csv("metadata/db_creds.csv")

# model params
similarity_top_k = 4
n_gpu_layers = 100
temperature = 0.0
max_new_tokens = 512
context_window = 3900
chunk_size = 512


def initialize(
    which_llm_local,
    which_corpus_local=None,
    n_gpu_layers=100,
    temperature=0.0,
    max_new_tokens=512,
    context_window=3900,
    chunk_size=512,
):
    text_path = (
        corpora_dict.loc[lambda x: x.name == which_corpus_local, "text_path"].values[0]
        if which_corpus_local is not None
        else None
    )
    metadata_path = (
        corpora_dict.loc[
            lambda x: x.name == which_corpus_local, "metadata_path"
        ].values[0]
        if which_corpus_local is not None
        else None
    )

    model = local_llm.local_llm(
        llm_url=llm_dict.loc[lambda x: x.name == which_llm_local, "llm_url"].values[0],
        llm_path=llm_dict.loc[lambda x: x.name == which_llm_local, "llm_path"].values[
            0
        ],
        redownload_llm=False,
        text_path=text_path,
        metadata_path=metadata_path,
        hf_token=os.getenv("HF_TOKEN"),
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
    )

    if which_corpus_local is not None:
        model.setup_db(
            user=db_info.loc[0, "user"],
            password=db_info.loc[0, "password"],
            table_name=which_corpus_local,
        )

        model.populate_db(chunk_size=chunk_size)
    return model, which_llm_local, which_corpus_local


@bot.message_handler(commands=["start", "hello"])
def send_welcome(message):
    bot.send_message(
        message.chat.id, text="Initializing, this may take a few minutes..."
    )

    if message.from_user.id in admins:
        if not ("model" in globals()):
            global model
            global which_llm
            global which_corpus
            model, which_llm, which_corpus = initialize(
                which_llm_local=llm_dict.loc[0, "name"],
                which_corpus_local=None,
                n_gpu_layers=n_gpu_layers,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
                chunk_size=chunk_size,
            )
        contextualized_section = " contextualized on the '{which_corpus}' corpus" if which_corpus is not None else ", not contextualized"
        bot.reply_to(
            message,
            f"""Successfully initialized! You are chatting with '{which_llm}'{contextualized_section}. 
Model parameters are: n_gpu_layers = {n_gpu_layers}, temperature = {temperature}, max_new_tokens = {max_new_tokens}, context_window = {context_window}, similarity_top_k = {similarity_top_k}, chunk_size = {chunk_size}.
If you want to include source documents in responses, write 'cite your sources' anywhere in the same message as your query. 
If you want to change the model and context, send a message in exactly this format: "[reinitialize]{{'new_llm':'llama-2-7b', 'new_corpus':'imf'}}"
The options for 'new_llm' are one of: {list(llm_dict.name)}
The options for 'new_corpus' are one of: {list(corpora_dict.name)}', or put None (no quotes) for non-RAG base model
            """,
        )
    else:
        bot.reply_to(message, "Unauthorized user")


# bot
@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    # check for reinitialization
    if "[reinitialize]" in message.text:
        # kill the existing conncection
        global model
        global which_corpus
        if which_corpus is not None:
            model.close_connection()
        del model.llm
        gc.collect()

        # params of the new model
        param_dict = eval(message.text.split("]")[1])
        new_corpus = param_dict["new_corpus"]
        new_llm = param_dict["new_llm"]
        bot.send_message(
            message.chat.id,
            text=f"Reinitializing the '{new_llm}' model on the '{new_corpus}' corpus, this may take a few minutes...",
        )

        global similarity_top_k
        global n_gpu_layers
        global temperature
        global max_new_tokens
        global context_window
        global chunk_size
        global which_llm

        similarity_top_k = (
            param_dict["similarity_top_k"]
            if "similarity_top_k" in param_dict.keys()
            else similarity_top_k
        )
        n_gpu_layers = (
            param_dict["n_gpu_layers"]
            if "n_gpu_layers" in param_dict.keys()
            else n_gpu_layers
        )
        temperature = (
            param_dict["temperature"]
            if "temperature" in param_dict.keys()
            else temperature
        )
        max_new_tokens = (
            param_dict["max_new_tokens"]
            if "max_new_tokens" in param_dict.keys()
            else max_new_tokens
        )
        context_window = (
            param_dict["context_window"]
            if "context_window" in param_dict.keys()
            else context_window
        )
        chunk_size = (
            param_dict["chunk_size"]
            if "chunk_size" in param_dict.keys()
            else chunk_size
        )

        model, which_llm, which_corpus = initialize(
            which_llm_local=new_llm,
            which_corpus_local=new_corpus,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            chunk_size=chunk_size,
        )

        response_message = (
            f"Successfully initialized! You are chatting with '{which_llm}' contextualized on the '{which_corpus}' corpus. Model parameters are: n_gpu_layers = {n_gpu_layers}, temperature = {temperature}, max_new_tokens = {max_new_tokens}, context_window = {context_window}, similarity_top_k = {similarity_top_k}, chunk_size = {chunk_size}"
            if new_corpus is not None
            else f"Successfully initialized! You are chatting with '{which_llm}', not contextualized. Model parameters are: n_gpu_layers = {n_gpu_layers}, temperature = {temperature}, max_new_tokens = {max_new_tokens}, context_window = {context_window}, chunk_size = {chunk_size}"
        )
        bot.send_message(message.chat.id, text=response_message)
    else:
        bot.send_message(
            message.chat.id,
            text=f"Thinking (model = '{which_llm}', corpus = '{which_corpus}')...",
        )

        try:
            response = model.gen_response(
                message.text.replace("cite your sources", ""),
                similarity_top_k=similarity_top_k,
            )
            # in case non-RAG
            if type(response) is not dict:
                response = {"response": response}
            bot.reply_to(message, response["response"])

            if "cite your sources" in message.text:
                bot.send_message(
                    message.chat.id, "These are the documents the reply is based on:"
                )

                for j in list(
                    pd.Series(list(response.keys()))[
                        pd.Series(list(response.keys())) != "response"
                    ]
                ):
                    bot.send_message(message.chat.id, f"{j}: " + response[j])
        except:
            bot.reply_to(
                message,
                "Context too large, try reformulating or shortening your question and asking again.",
            )


bot.infinity_polling()
