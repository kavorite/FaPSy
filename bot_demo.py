import asyncio
import dataclasses
import os
import pickle
from collections import deque
from shutil import rmtree
from typing import Awaitable, Callable, Dict, List, Union
from urllib.request import urlopen

import aiohttp
import numpy as np
import pyrogram as tg
import river as rv
from expiringdict import ExpiringDict

import pixie
from common import PostGraph

with urlopen("http://checkip.amazonaws.com/") as rsp:
    remote_host = rsp.read().decode("utf8")[:-1]
bot = tg.Client(session_name=remote_host, bot_token=os.environ["BOT_TOKEN"])
graph = PostGraph("./index/")


@dataclasses.dataclass(frozen=True, eq=True)
class CBKey:
    user_id: int
    op_name: str


@dataclasses.dataclass(frozen=True, eq=True)
class CBCfg:
    handler: Callable[[tg.types.CallbackQuery], Awaitable[None]]
    prompt: str


class QueryDispatcher:
    ledger = dict()

    def __init__(self):
        self.ledger = self.__class__.ledger

    def register(self, key: CBKey, cfg: CBCfg):
        self.ledger[key] = cfg

    def inline_prompt(
        self,
        prompt_rows: Union[List[Dict[str, CBCfg]], Dict[str, CBCfg]],
        reply_to: tg.types.Message,
    ):
        if not isinstance(prompt_rows, List):
            prompt_rows = [prompt_rows]
        button_rows = []
        for row in prompt_rows:
            button_row = []
            for op_name, cfg in row.items():
                key = CBKey(reply_to.from_user.id, op_name)
                button_row.append(
                    tg.types.InlineKeyboardButton(
                        text=cfg.prompt, callback_data=key.op_name
                    )
                )
                self.register(key, cfg)
            button_rows.append(button_row)
        return tg.types.InlineKeyboardMarkup(inline_keyboard=button_rows)

    async def dispatch(self, client: tg.Client, query: tg.types.CallbackQuery):
        key = CBKey(query.from_user.id, query.data)
        if key in self.ledger:
            cfg = self.ledger.pop(key)

            async def try_op():
                if callable(cfg.handler):
                    await cfg.handler(query)

            asyncio.create_task(try_op())


class Recommender:
    def __init__(
        self, save_path, model: rv.ensemble.AdaptiveRandomForestClassifier = None
    ):
        if model is None:
            model = rv.ensemble.AdaptiveRandomForestClassifier()
        self.seen = deque(maxlen=32)
        self.model = model
        self.save_path = save_path

    POSITIVE = 1
    NEGATIVE = 0

    @staticmethod
    def _prep(query):
        query = np.array(query)
        mag = np.array(np.linalg.norm(query))
        mag[mag == 0] = 1.0
        inv = 1 / mag
        return rv.utils.numpy2dict(query * inv)

    def train(self, query, positive):
        x = self._prep(query)
        y = self.__class__.POSITIVE if positive else self.__class__.NEGATIVE
        self.model.learn_one(x, y)
        self.seen.add(query.tobytes())

    def infer(self, query):
        if query.tobytes() in self.seen:
            return 0.1
        yhat = self.model.predict_proba_one(rv.utils.numpy2dict(query))
        if yhat is None:
            return 0.5
        return yhat[self.__class__.POSITIVE]

    def save(self, path=None):
        with open(path or self.save_path, "wb+") as ostrm:
            pickle.dump(self, ostrm)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as istrm:
            return pickle.load(istrm)

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        self.save()


class Preferences:
    def __init__(
        self,
        ident: tg.types.User,
        save_root: str,
        model: Recommender,
    ):
        self.save_root = save_root
        self.ident = ident
        self.model = model

    def model_path(save_root, user_id):
        return os.path.join(save_root, f"{user_id}.visit.npz")

    @classmethod
    def load_else_init(cls, save_root: str, ident: tg.types.User):
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        prefs_path = cls.model_path(save_root, ident.id)
        if os.path.exists(prefs_path):
            model = Recommender.load(prefs_path)
        else:
            model = Recommender(save_path=prefs_path)
        return cls(ident, save_root, model)

    def infer_rating(self, query):
        return self.model.infer(query)

    def learn_rating(self, query, is_positive=True):
        with self.model:
            query = np.array(query)
            self.model.train(query, is_positive)


class PrefManager:
    def __init__(self, save_root):
        self.save_root = os.path.abspath(save_root)
        # only cache user data up to the first power of two over 100k for up to a
        # kibisecond
        self.cache = ExpiringDict(max_age_seconds=1 << 10, max_len=1 << 17)

    def save_path(self, ident: tg.types.User):
        return os.path.join(self.save_root, str(ident.id))

    def prefs_for(self, ident: tg.types.User) -> Preferences:
        if ident.id in self.cache:
            return self.cache[ident.id]
        else:
            prefs = Preferences.load_else_init(self.save_path(ident), ident)
            self.cache[ident.id] = prefs
            return prefs

    def drop(self, ident: tg.types.User):
        try:
            rmtree(self.save_path(ident))
        except FileNotFoundError:
            pass
        if ident.id in self.cache:
            del self.cache[ident.id]


query_callbacks = QueryDispatcher()
user_preferences = PrefManager("./fapsy.cache")


@bot.on_callback_query()
async def dispatch_cb_query(client: tg.Client, query: tg.types.CallbackQuery):
    asyncio.create_task(query_callbacks.dispatch(client, query))


@bot.on_message(filters=tg.filters.command(["start"]))
async def welcome(_: tg.Client, msg: tg.types.Message):
    status_paras = """
        Welcome! To start, send me one or more illustrated images of 
        anthropomorphic characters, or send me a list of tags with /avoid or
        /enjoy `space_delimited search terms`, and I'll use them to begin
        curating  your selection. You can send me more any time you want, I'll
        do my best to incorporate your feedback 😊

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
        await query.answer("Dropping preferences...")
        user_preferences.drop(query.from_user)
        await client.send_message(msg.from_user.id, "☠ Bombs away. Have a good one.")
        drop_cbs = []
        for key in query_callbacks.ledger.keys():
            if query.from_user.id == msg.from_user.id:
                drop_cbs.append(key)
        for drop in drop_cbs:
            del query_callbacks.ledger[drop]

    async def on_cancel(_: tg.types.CallbackQuery):
        await client.send_message(client.from_user.id, "Data kept.")

    markup = query_callbacks.inline_prompt(
        {
            "drop_data": CBCfg(on_confirm, "Okay."),
            "keep_data": CBCfg(on_cancel, "Don't."),
        },
        reply_to=msg,
    )
    await msg.reply(status, reply_markup=markup)


@bot.on_message(filters=tg.filters.command(["about"]))
async def send_about(client: tg.Client, msg: tg.types.Message):
    author, *_ = await client.get_users([369922275])
    status = (
        f"[FaPSy](https://github.com/kavorite/FaPSy) is from"
        f" [{author.first_name}](tg://resolve?domain={author.username})"
        r" with 💕"
    )
    await msg.reply(status, parse_mode="markdown")


async def index_query(msg: tg.types.Message, positive):
    # TODO: auto-correct
    _, *given = msg.text.split()
    given = set(given)
    known = given.intersection(set(graph.attenuator.V.keys()))
    unknown = given - known
    status = f"I don't recognize these tags: `{' '.join(unknown)}`."
    if len(unknown) == len(given):
        status += " No known tags provided. Aborting."

    if unknown:
        await msg.reply(status, parse_mode="markdown")
    else:
        reply = await msg.reply("indexing...")
        prefs = user_preferences.prefs_for(msg.from_user)
        q = graph.attenuator(" ".join(known))
        prefs.learn_rating(q, positive)
        for t in known:
            prefs.learn_rating(graph.attenuator(t), positive)
        await reply.edit_text("...indexed successfully 😊")


@bot.on_message(filters=tg.filters.command(["enjoy"]))
async def index_positive(_: tg.Client, msg: tg.types.Message):
    await index_query(msg, True)


@bot.on_message(filters=tg.filters.command(["avoid"]))
async def index_positive(_: tg.Client, msg: tg.types.Message):
    await index_query(msg, False)


@bot.on_message(filters=tg.filters.photo)
async def index_photo(client: tg.Client, msg: tg.types.Message):
    reply = await msg.reply("indexing...")
    if msg.photo.thumbs:
        photo = sorted(msg.photo.thumbs, key=lambda thumb: thumb.file_size)[0]
    else:
        photo = msg.photo
    handle = tg.file_id.FileId.decode(photo.file_id)
    location = tg.raw.types.InputPhotoFileLocation(
        id=handle.media_id,
        access_hash=handle.access_hash,
        file_reference=handle.file_reference,
        thumb_size=handle.thumbnail_size,
    )
    img_str = bytearray()
    cursor = 0
    while cursor < photo.file_size - 1:
        fetch = tg.raw.functions.upload.GetFile(
            location=location,
            cdn_supported=False,
            offset=0,
            limit=1 << 20,
        )
        rsp = await client.send(fetch)
        cursor += len(rsp.bytes)
        img_str += rsp.bytes
        if not rsp.bytes:
            break
    img_str = rsp.bytes
    query = graph.recognizer(img_str)
    prefs = user_preferences.prefs_for(msg.from_user)
    prefs.learn_rating(query, True)
    await reply.edit_text("...indexed successfully 😊")


@bot.on_message(tg.filters.command(["walk"]))
async def walk_step(
    client: tg.Client, msg: tg.types.Message, last_vibecheck=None, liked=None
):
    reply = await msg.reply("searching...")
    prefs = user_preferences.prefs_for(msg.from_user)

    def query_neighbors(q):
        max_degree = 16
        search_params = dict(n=max_degree, search_k=max_degree * 2)
        if isinstance(q, bytes):
            return graph.index.get_nns_by_vector(np.frombuffer(q), **search_params)
        else:
            return graph.index.get_nns_by_item(q, **search_params)

    def query_user_pref(query):
        if isinstance(query, bytes):
            q = np.frombuffer(query)
        else:
            q = np.array(graph.index.get_item_vector(query))
        return prefs.infer_rating(q)

    traversal = pixie.Traversal(
        query_neighbors,
        query_user_pref,
        tour_hops=4,
        node_goal=4,
        goal_hits=8,
        max_steps=128,
    )
    if liked:
        search = [liked]
    else:
        search = list(traversal.rng.integers(0, graph.index.get_n_items(), size=(8,)))
    visits, counts = zip(*pixie.random_walk(search, traversal).items())
    counts = np.array(counts)
    visits = np.array(visits, dtype=object)
    node_id = visits[np.argmax(counts)]
    post_id = node_id + graph.offset
    query = graph.index.get_item_vector(node_id)
    endpoint = f"https://e621.net/posts/{post_id}"
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

    async def rate_if_hot(cbq: tg.types.CallbackQuery):
        prefs.learn_rating(query, 0)
        asyncio.create_task(cbq.answer())
        asyncio.create_task(walk_step(client, msg, vibecheck))

    async def rate_if_not(cbq: tg.types.CallbackQuery):
        prefs.learn_rating(query, 1)
        asyncio.create_task(cbq.answer())
        asyncio.create_task(walk_step(client, msg, vibecheck, liked=query))

    async def end_tour(query: tg.types.CallbackQuery):
        await query.answer("All right, I learned a lot! Message me again sometime.")

    rating_prompt = query_callbacks.inline_prompt(
        {
            "rate_not": CBCfg(rate_if_not, "👎"),
            "end_tour": CBCfg(end_tour, "🛑"),
            "rate_hot": CBCfg(rate_if_hot, "👍"),
        },
        reply_to=msg,
    )
    status = f"{endpoint}\n{vibecheck}"
    reply = await reply.edit_text(status, reply_markup=rating_prompt)


async def main():
    await bot.start()
    commands = {
        "about": "about the author 💕",
        "start": "send a help and welcome message",
        "help": "send a help and welcome message",
        "walk": "begin a new tour",
        "enjoy": "specify a tag combination you like",
        "avoid": "specify a tag combination you dislike",
        "stop": "halt and drop user data",
    }
    await bot.send(
        tg.raw.functions.bots.SetBotCommands(
            commands=[
                tg.raw.types.BotCommand(command=cmd, description=dsc)
                for cmd, dsc in commands.items()
            ]
        )
    )
    await tg.idle()
    await bot.stop()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
