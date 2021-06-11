import asyncio
import dataclasses
import io
import os
from collections import deque
from typing import Awaitable, Callable, Dict, Iterable, Optional, Union

import aiohttp
import numpy as np
import pyrogram as tg
import tensorflow as tf
from expiringdict import ExpiringDict

import pixie
from common import PostGraph

bot = tg.Client(os.environ["BOT_TOKEN"])
graph = PostGraph.load_path(".")
tau = (np.sqrt(5) + 1) / 2


@dataclasses.dataclass
class CBKey:
    user_id: int
    op_name: str


@dataclasses.dataclass
class CBCfg:
    handler: Callable[[tg.types.CallbackQuery], Awaitable[None]]
    prompt: str
    reply_to: Optional[tg.types.Message]


class QueryDispatcher:
    ledger = dict()

    def __init__(self):
        self.ledger = self.__class__.ledger

    def register(self, key: CBKey, cfg: CBCfg):
        self.ledger[key] = cfg

    def inline_prompt(
        self,
        reply_to: tg.types.Message,
        prompt_rows: Union[Iterable[Dict[str, CBCfg]], Dict[str, CBCfg]],
    ):
        if not isinstance(prompt_rows, Iterable):
            prompt_rows = (prompt_rows,)
        button_rows = []
        for row in prompt_rows:
            button_row = []
            for op_name, cfg in enumerate(row.items()):
                key = CBKey(reply_to.from_user.id, op_name)
                button_row.append(
                    tg.types.InlineKeyboardButton(cfg.prompt, callback_data=key.op_name)
                )
                self.register(key, dataclasses.replace(cfg, reply_to=reply_to))
            button_rows.append(button_row)
        return tg.types.InlineKeyboardMarkup(button_rows)

    def dispatch(self, client: tg.Client, query: tg.types.CallbackQuery):
        key = CBKey(query.from_user.id, query.data)
        if key in self.ledger:
            cfg = self.ledger.pop(key)

            async def try_op():
                try:
                    if callable(cfg.handler):
                        await cfg.handler(query)
                except Exception as err:
                    status = (
                        f"Error during <code>{key.op_name}</code>. Backtrace attached."
                    )
                    backtrace = io.BytesIO(str(err).encode("utf8"))
                    setattr(backtrace, "name", "backtrace.txt")
                    await client.send_document(
                        query.from_user,
                        status,
                        document=backtrace,
                        parse_mode="html",
                        reply_to=cfg.reply_to,
                    )

            asyncio.create_task(try_op())


class QueryRegressor:
    @staticmethod
    def build_model(A, q):
        P = tf.linalg.lstsq(A, q)
        x = tf.linalg.lstsq(P @ q, 1.57)
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
        return model

    def __init__(self, graph: PostGraph, query, save_path):
        # TODO(kavorite): get rid of blocking I/O
        if os.path.exists(save_path):
            self.model = tf.keras.models.load_model(save_path)
        else:
            self.model = self.build_model(graph.embed.A, query)
        self.save_path = save_path

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        # TODO(kavorite): get rid of blocking I/O
        self.model.save(self.save_path)

    def train(self, query, rating):
        with self:
            self.model.train_on_batch(query, rating)

    def infer(self, query):
        return self.model(query)


class RatingTape:
    def __init__(self, save_path, max_len=32, alpha=tau / 2):
        self.save_path = save_path
        self.vecs = deque(max_len)
        self.tape = deque(max_len)
        self.alpha = alpha

    def train(self, query, rating):
        self.vecs.append(query)
        self.norms.append(np.linalg.norm(query + 1e-16))
        self.tape.append(rating)

    def infer(self, query):
        for i, p, s in enumerate(zip(self.vecs, self.tape)):
            if np.allclose(query, p):
                s *= self.alpha ** (len(self.ratings) - i - 1)
                return s
        return None

    def likes(self):
        for q, s in zip(self.vecs, self.tape):
            if s > 0.5:
                yield q

    def save(self, path=None):
        # TODO(kavorite): get rid of blocking I/O
        index = np.array(self.index)
        ratings = np.array(self.ratings)
        max_len = np.array(self.ratings.max_len)
        alpha = np.array(self.alpha)
        np.savez(
            self.save_path or path,
            alpha=alpha,
            index=index,
            ratings=ratings,
            max_len=max_len,
            allow_pickle=False,
        )

    @classmethod
    def load(cls, save_path):
        # TODO(kavorite): get rid of blocking I/O
        dictfile = np.load(save_path)
        self = cls(save_path, dictfile["max_len"])
        self.index = deque(dictfile["index"])
        self.ratings = deque(dictfile["ratings"])
        return self

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        self.save(self, self.save_path)


class Preferences:
    def __init__(
        self,
        ident: tg.types.User,
        save_root: str,
        rater: QueryRegressor,
        visit: RatingTape,
    ):
        self.save_root = save_root
        self.ident = ident
        self.rater = rater
        self.visit = visit

    @staticmethod
    def rater_path(save_root, user_id):
        return os.path.join(save_root, f"{user_id}.prefs.h5")

    def visit_path(save_root, user_id):
        return os.path.join(save_root, f"{user_id}.visit.npz")

    @classmethod
    def load(cls, save_root: str, ident: tg.types.User):
        visit_path = cls.rater_path(save_root, ident.id)
        rater_path = cls.visit_path(save_root, ident.id)
        if not os.path.exists(visit_path):
            raise FileNotFoundError(
                f"{save_root} contains no rating tape at {visit_path}"
            )
        if not os.path.exists(rater_path):
            raise FileNotFoundError(
                f"{save_root} contains no embedding regressor at {rater_path}"
            )
        rater = QueryRegressor.load(rater_path)
        visit = RatingTape.load(visit_path)
        return cls(ident, save_root, rater, visit)

    def infer_rating(self, query):
        found = self.visit.get_rating(query)
        if found:
            return found

        return self.rater.infer(query)

    def learn_rating(self, query, rating):
        with self.visit:
            self.visit.train(query, rating)
        with self.rater:
            self.rater.train(query, rating)


class PrefManager:
    def __init__(self, save_root):
        self.save_root = os.path.abspath(save_root)
        # only cache user data up to the first power of two over 100k for up to a
        # kibisecond
        self.cache = ExpiringDict(max_age_seconds=1 << 10, max_len=1 << 17)

    @staticmethod
    def uid(ident: Union[tg.types.User, int]):
        if isinstance(ident, tg.types.User):
            return ident.id
        else:
            return ident

    def prefs_for(self, ident: tg.types.User) -> Preferences:
        uid = self.uid(ident)
        if uid in self.cache:
            return self.cache[uid]
        else:
            prefs = Preferences.load(os.path.join(self.save_root, str(ident.id)))
            self.cache[ident.id] = prefs
            return prefs


query_callbacks = QueryDispatcher()
user_preferences = PrefManager()


@bot.on_callback_query
async def dispatch_cb_query(client: tg.Client, query: tg.types.CallbackQuery):
    query_callbacks.dispatch(client, query)


@bot.on_message(filters=tg.filters.command(["start"]))
async def welcome(_: tg.Client, msg: tg.types.Message):
    status_paras = """
        Welcome! To start, send me one or more illustrated images of 
        anthropomorphic characters, or send me a list of tags with /rate
        `space_delimited search terms`, and I'll use them to begin curating 
        your selection. You can send me more any time you want, I'll do my
        best to incorporate your feedback 😊

        Once you've sent me one or two images, I can start personalizing your
        experience, and you can begin a new tour with /walk.
        """.split(
        "\n\n"
    )
    status = "\n\n".join(" ".join(p.split()) for p in status_paras)
    await msg.reply(status)


@bot.on_message(filters=tg.filters.command(["stop"]))
async def confirm_drop(client: tg.Client, msg: tg.types.Message):
    status = " ".join(
        """
        This will delete all of our records on your preferences. Please
        confirm.
        """.split()
    )

    async def on_confirm(query: tg.types.CallbackQuery):
        status = "Dropping preferences..."
        reply = await client.send_message(query.from_user.id, status)
        drop = user_preferences.prefs_for(query.from_user.id)
        if os.path.exists(drop):
            os.remove(drop)
        await reply.edit_text("☠ Bombs away. Have a good one.")
        drop_cbs = []
        for key in query_callbacks.ledger.keys():
            if drop.user_id == msg.from_user.id:
                drop_cbs.append(key)
        for drop in drop_cbs:
            del query_callbacks.ledger[drop]

    async def on_cancel(_: tg.types.CallbackQuery):
        await client.send_message(client.from_user.id, "Data kept.")

    markup = query_callbacks.inline_prompt(
        {
            "drop_data": CBCfg(on_confirm, "⚠ Please drop my data."),
            "keep_data": CBCfg(on_cancel, "✓ Please keep my data."),
        }
    )
    await msg.reply(status, reply_markup=markup)


@bot.on_message(filters=tg.filters.command(["about"]))
async def send_about(client: tg.Client, msg: tg.types.Message):
    author, *_ = await client.get_users([369922275])
    status = (
        f'from <a href="tg://resolve?domain={author.username}">'
        f"{author.first_name}</a> with 💕"
    )
    await msg.reply(status, parse_mode="html")


async def index_query(msg: tg.types.Message, rating):
    reply = await msg.reply("indexing...")
    q = graph.embed(msg.text)
    prefs = user_preferences.prefs_for(msg.from_user)
    prefs.learn_rating(q, rating)
    await reply.edit_text("...indexed successfully 😊")


@bot.on_message(filters=tg.filters.command(["enjoy"]))
async def index_positive(_: tg.Client, msg: tg.types.Message):
    await index_query(msg, 1.0)


@bot.on_message(filters=tg.filters.command(["avoid"]))
async def index_positive(_: tg.Client, msg: tg.types.Message):
    await index_query(msg, 0.0)


@bot.on_message(filters=tg.filters.photo)
async def index_photo(client: tg.Client, msg: tg.types.Message):
    reply = await msg.reply("...indexed successfully 😊")
    if msg.photo.thumbs:
        photo = sorted(msg.photo.thumbs, key=lambda thumb: thumb.file_size)[0]
    else:
        photo = msg.photo
    handle = tg.file_id.FileId.decode(photo.file_id)
    location = tg.raw.functions.upload.InputPhotoFileLocation(
        id=handle.media_id,
        access_hash=handle.access_hash,
        file_reference=handle.file_reference,
        thumb_size=handle.thumbnail_size,
    )
    cursor = 0
    img_str = bytearray()
    while True:
        fetch = tg.raw.functions.upload.GetFile(
            location, offset=cursor, limit=photo.file_size
        )
        rsp = await client.send(fetch)
        if isinstance(rsp, tg.raw.types.upload.File):
            chunk = rsp.bytes
            img_str += chunk
            if not chunk:
                break
    img_str = bytes(img_str)
    query = graph.embed(img_str)
    prefs = user_preferences.prefs_for(msg.from_user)
    prefs.learn_rating(query, 1.0)
    await reply.edit_text("...indexed successfully 😊")


@bot.on_message(tg.filters.command(["walk"]))
async def walk_step(client: tg.Client, msg: tg.types.Message, last_vibecheck=None):
    prefs = user_preferences.prefs_for(msg.from_user)

    def query_neighbors(query):
        return [graph.index.get_item_vector(i) for i in graph.neighbors(query)]

    def query_user_pref(query):
        return prefs.infer_rating(query)

    reply = await msg.reply("searching...")
    traversal = pixie.Traversal(query_neighbors, query_user_pref)
    likes = list(prefs.visit.likes())
    tour = pixie.random_walk(likes, traversal)
    visits, counts = zip(*tour)
    counts = np.array(counts)
    visits = np.array(visits, dtype=object)
    query = visits[np.argmax(counts)]
    post_id = graph.neighbors(query)[0]
    async with aiohttp.ClientSession() as http:
        endpoint = f"https://e621.net/posts/{post_id}.json"
        async with http.get(
            endpoint,
            headers={"User-Agent": "e6tag by https://github.com/kavorite"},
        ) as rsp:
            post = await rsp.json()
            if "success" in post and not post["success"]:
                status = (
                    f"fetch [/posts/{post_id}]({endpoint[:-5]}): " f"{post['reason']}"
                )
                await reply.edit_text(status, parse_mode="markdown")
                return
    await client.send(
        tg.raw.messages.SendMedia(
            tg.raw.types.InputMediaDocumentExternal(post["sample"]["url"]),
            message=endpoint[:-5],
            peer=await client.resolve_peer(msg.from_user.id),
            reply_to_msg_id=reply.id,
            random_id=tg.session.internals.MsgId(),
        )
    )
    vibechecks = [
        "Am I good? Or am I good?",
        "I'm feeling lucky.",
        "Let's push the envelope.",
        "Okay, show me the money.",
        "Pretty sure I nailed it.",
        "This should be good.",
    ]
    if last_vibecheck is not None:
        vibechecks.pop(np.searchsorted(vibechecks, last_vibecheck))
    vibecheck = np.random.choice(vibechecks)
    await reply.edit_text(vibecheck)

    async def rate_if_hot(_: tg.types.CallbackQuery):
        prefs.learn_rating(query, 1.0)
        asyncio.create_task(walk_step(client, msg, vibecheck))

    async def rate_if_not(_: tg.types.CallbackQuery):
        prefs.learn_rating(query, 0.0)
        asyncio.create_task(walk_step(client, msg, vibecheck))

    async def end_tour(_: tg.types.CallbackQuery):
        await reply.edit_text("All right, I learned a lot! Visit me again some time 😊")

    await reply.edit_reply_markup(
        query_callbacks.inline_prompt(
            {
                "rate_not": CBCfg(rate_if_not, "◀"),
                "end_tour": CBCfg(end_tour, "■"),
                "rate_hot": CBCfg(rate_if_hot, "▶"),
            },
        )
    )


@bot.on_message(filters=tg.filters.command(["walk"]))
async def start_tour(_: tg.types.Client, msg: tg.types.Message):
    if not os.path.exists(user_preferences.model_path(msg.from_user.id)):
        status = " ".join(
            """
            Looks like we don't have any data on your preferences! Send me a
            post or two, then start a tour with /walk 😊"
            """.split()
        )
        await msg.reply(status)
        return


bot.send(
    tg.raw.functions.bots.SetBotCommands(
        [
            tg.raw.types.BotCommand("start", "send a help and welcome message"),
            tg.raw.types.BotCommand("about", "about the author 💕"),
            tg.raw.types.BotCommand("walk", "begin a new tour"),
            tg.raw.types.BotCommand("enjoy", "tell me a tag combination you like"),
            tg.raw.types.BotCommand("avoid", "tell me a tag combination you dislike"),
            tg.raw.types.BotCommand("stop", "halt and drop user data"),
        ]
    )
)
bot.start()
