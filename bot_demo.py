import asyncio
import io
import os
from typing import Callable, Dict, Tuple

import pyrogram as tg
import tensorflow as tf
from expiringdict import ExpiringDict

from common import PostGraph

bot = tg.Client(os.environ["BOT_TOKEN"])
graph = PostGraph.load_path(".")


class UserID(int):
    pass


class OpName(str):
    pass


CallbackHandler = Callable[[tg.types.CallbackQuery], None]

query_callbacks: Dict[
    Tuple[UserID, OpName], Tuple[CallbackHandler, tg.types.Message]
] = dict()
filter_cache = ExpiringDict(max_age_seconds=1024, max_len=1 << 20)


def register_query_cb(
    user_id, op_name, handler: CallbackHandler, reply_to: tg.types.Message = None
):
    query_callbacks[(user_id, op_name)] = (handler, reply_to)


class TagFilter:
    @staticmethod
    def build_model(A, q):
        P = tf.linalg.lstsq(A, q)
        x = tf.linalg.lstsq(P @ q.reshape(len(q), 1), 1.57)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=graph.embed.n_dim,
                    kernel_initializer=tf.keras.initializers.Constant(P),
                    name="personalization",
                ),
                tf.keras.layers.Dense(
                    units=1,
                    activation="swish",
                    name="preference",
                    kernel_initializer=tf.keras.initializers.Constant(x),
                ),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )
        model.train_step()
        return model

    @classmethod
    def init(cls, graph: PostGraph, tags, save_path):
        model = cls.build_model(graph.embed.A, graph.embed(tags))
        return cls(model, save_path)

    def load(self, save_path):
        self.model = tf.keras.models.load_model(save_path)
        self.save_path = save_path
        return self

    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        self.model.save(self.save_path)


def filter_path(uid):
    return (f"./user_pref/{uid}",)


def filter_for(uid):
    if uid not in filter_cache:
        filter_cache[uid] = TagFilter(graph, filter_path(uid))
    return filter_cache[uid]


@bot.on_message(filters=tg.filters.command(["start"]))
async def welcome(_: tg.Client, msg: tg.types.Message):
    status_paras = """
        Welcome! To start, send me one or more images from E621, and I'll use
        them to begin curating your selection. You can send me more any time
        you want, I'll do my best to incorporate your feedback 😊

        Once you've sent me one or two images, I can start personalizing your
        experience, and you can begin a new tour with /walk.
        """.split(
        "\n\n"
    )
    status = "\n\n".join(" ".join(p.split()) for p in status_paras)
    await msg.reply(status)
    if not os.path.exists(filter_path(msg.from_user.id)):
        status = " ".join(
            """
            Looks like we don't have any data on your preferences! Send me a
            post or two, then start a tour with /walk 😊"
            """.split()
        )
        await msg.reply(status)
    else:
        await start_tour()


@bot.on_message(filters=tg.filters.command(["stop"]))
async def confirm_drop(client: tg.Client, msg: tg.types.Message):
    status = " ".join(
        """
        This will delete all of our records on your preferences. Please
        confirm.
        """.split()
    )
    await msg.reply(
        status,
        reply_markup=tg.types.ReplyKeyboardMarkup(
            [
                [
                    tg.types.InlineKeyboardButton("Please keep my data."),
                    tg.types.InlineKeyboardButton(
                        "Please drop my data.", callback_data="drop_prefs"
                    ),
                ]
            ],
            one_time_keyboard=True,
            selective=True,
        ),
    )

    async def on_confirm(query: tg.types.CallbackQuery):
        status = "Dropping preferences..."
        reply = await client.send_message(query.from_user.id, status)
        try:
            drop = filter_path(query.from_user.id)
            if os.path.exists(drop):
                os.remove(drop)
            await reply.edit_text("Bombs away. Have a good one.")
        except Exception as err:
            await reply.edit_text(f"error: {err}")

    register_query_cb(msg.from_user.id, "drop_prefs", reply_to=msg, handler=on_confirm)


@bot.on_message(filters=tg.filters.command(["about"]))
async def send_about(client: tg.Client, msg: tg.types.Message):
    author, *_ = await client.get_users([369922275])
    status = (
        f'from <a href="tg://resolve?domain={author.username}">'
        f"{author.first_name}</a> with 💕"
    )
    await msg.reply(status, parse_mode="html")


@bot.on_message(filters=tg.filters.command(["walk"]))
async def start_tour():
    pass


@bot.on_callback_query
async def drop_prefs(client: tg.Client, query: tg.types.CallbackQuery):
    op_name = query.data
    key = (query.from_user.id, op_name)
    client.get_history()
    client.get_chat(query.from_user.id)
    if key in query_callbacks:
        handler, reply_to = query_callbacks.pop(key)

        async def try_op():
            try:
                await handler(query)
            except Exception as err:
                status = f"Error during <code>{op_name}</code>. Backtrace attached."
                backtrace = io.BytesIO(str(err).encode("utf8"))
                setattr(backtrace, "name", "backtrace.txt")
                await client.send_document(
                    query.from_user,
                    status,
                    document=backtrace,
                    parse_mode="html",
                    reply_to=reply_to,
                )

        asyncio.create_task(try_op)


bot.send(
    tg.raw.functions.bots.SetBotCommands(
        [
            tg.raw.types.BotCommand("start", "send a help and welcome message"),
            tg.raw.types.BotCommand("about", "about the author 💕"),
            tg.raw.types.BotCommand("walk", "begin a new tour"),
            tg.raw.types.BotCommand("stop", "halt and drop user data"),
        ]
    )
)
bot.start()
