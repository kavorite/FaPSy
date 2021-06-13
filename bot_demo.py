import asyncio
import dataclasses
import os
from collections import deque
from typing import Awaitable, Callable, Dict, List, Union
from urllib.request import urlopen

import aiohttp
import numpy as np
import pyrogram as tg
from expiringdict import ExpiringDict

import pixie
from common import PostGraph

with urlopen("http://checkip.amazonaws.com/") as rsp:
    remote_host = rsp.read().decode("utf8")[:-1]
bot = tg.Client(session_name=remote_host, bot_token=os.environ["BOT_TOKEN"])
graph = PostGraph("./index/")
tau = (np.sqrt(5) + 1) / 2


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


class RatingTape:
    def __init__(
        self, save_path, max_len=64, alpha=tau / 2, beta=2.0, rvecs=[], rtape=[]
    ):
        self.save_path = save_path
        self.rvecs = deque(rvecs, maxlen=int(max_len))
        self.rtape = deque(rtape, maxlen=int(max_len))
        self.alpha = alpha
        self.sigma = 1 / beta

    def train(self, query, rating):
        norm = np.linalg.norm(query)
        if norm != 0:
            query /= norm
        self.rvecs.appendleft(query)
        self.rtape.appendleft(rating)

    def infer(self, query, k=8):
        q = np.array(query)
        if len(q.shape) == 1:
            q = q[None, :]
        X = np.vstack(self.rvecs)
        s = self.alpha ** np.arange(len(self.rvecs))
        s += self.sigma
        s /= self.sigma + 1
        yhats = 1 - np.squeeze(q @ X.T)
        return sum(self.rtape[i] for i in np.argsort(yhats)[::-1][:k]) / k

    def likes(self):
        for q, s in zip(self.rvecs, self.rtape):
            if s > 0.5:
                yield q

    def save(self, path=None):
        # TODO(kavorite): get rid of blocking I/O
        rvecs = np.array(self.rvecs)
        rtape = np.array(self.rtape)
        max_len = np.array(self.rvecs.maxlen, dtype=np.int32)
        alpha = np.array(self.alpha)
        np.savez(
            self.save_path or path,
            alpha=alpha,
            rvecs=rvecs,
            rtape=rtape,
            max_len=max_len,
            allow_pickle=False,
        )

    @classmethod
    def load(cls, save_path):
        # TODO(kavorite): get rid of blocking I/O
        dictfile = np.load(save_path)
        slots = "rvecs", "rtape", "alpha", "max_len"
        self = cls(save_path, **{k: dictfile[k] for k in slots})
        return self

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        self.save()


class Preferences:
    def __init__(
        self,
        ident: tg.types.User,
        save_root: str,
        rtape: RatingTape,
    ):
        self.save_root = save_root
        self.ident = ident
        self.rtape = rtape

    def rtape_path(save_root, user_id):
        return os.path.join(save_root, f"{user_id}.visit.npz")

    @classmethod
    def load_else_init(cls, save_root: str, ident: tg.types.User):
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        rtape_path = cls.rtape_path(save_root, ident.id)
        if os.path.exists(rtape_path):
            rtape = RatingTape.load(rtape_path)
        else:
            rtape = RatingTape(rtape_path)
        return cls(ident, save_root, rtape)

    def infer_rating(self, query):
        return self.rtape.infer(query)

    def learn_rating(self, query, rating):
        with self.rtape:
            self.rtape.train(query, rating)


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
            save_root = os.path.join(self.save_root, str(ident.id))
            prefs = Preferences.load_else_init(save_root, ident)
            self.cache[ident.id] = prefs
            return prefs


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
        root = user_preferences.prefs_for(query.from_user).save_root
        if os.path.exists(root):
            os.rmdir(root)
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


async def index_query(msg: tg.types.Message, rating):
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
        q = graph.attenuator(" ".join(known))
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
    reply = await msg.reply("indexing...")
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
    query = graph.attenuator(img_str)
    prefs = user_preferences.prefs_for(msg.from_user)
    prefs.learn_rating(query, 1.0)
    await reply.edit_text("...indexed successfully 😊")


@bot.on_message(tg.filters.command(["walk"]))
async def walk_step(client: tg.Client, msg: tg.types.Message, last_vibecheck=None):
    prefs = user_preferences.prefs_for(msg.from_user)

    def query_neighbors(query):
        for q in graph.neighbors(query, 8, search_k=64):
            x = graph.index.get_item_vector(q)
            yield tuple(np.around(x, 3))

    def query_user_pref(query):
        return prefs.infer_rating(query)

    reply = await msg.reply("searching...")
    traversal = pixie.Traversal(query_neighbors, query_user_pref)
    likes = set(tuple(np.around(x, 3)) for x in prefs.rtape.likes())
    tour = pixie.random_walk(likes, np.ones(len(likes)), traversal)
    visits, counts = zip(*tour)
    counts = np.array(counts)
    visits = np.array(visits, dtype=object)
    query = visits[np.argmax(counts)]
    post_id = graph.neighbors(query)[0]
    async with aiohttp.ClientSession() as http:
        endpoint = f"https://e621.net/posts/{post_id}.json"
        async with http.get(
            endpoint,
            headers={"User-Agent": "https://github.com/kavorite/TaPSy"},
        ) as rsp:
            post = await rsp.json()
            if "success" in post and not post["success"]:
                status = (
                    f"fetch [/posts/{post_id}]({endpoint[:-5]}): " f"{post['reason']}"
                )
                await reply.edit_text(status, parse_mode="markdown")
                return
    media = tg.raw.types.InputMediaDocumentExternal(url=post["sample"]["url"])
    reply = await reply.edit_text(text=endpoint[:-5])
    reply = await reply.edit_media(media=media)
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

    async def rate_if_hot(_: tg.types.CallbackQuery):
        prefs.learn_rating(query, 1.0)
        await query.answer(vibecheck)
        asyncio.create_task(walk_step(client, msg, vibecheck))

    async def rate_if_not(_: tg.types.CallbackQuery):
        prefs.learn_rating(query, 0.0)
        await query.answer(vibecheck)
        asyncio.create_task(walk_step(client, msg, vibecheck))

    async def end_tour(_: tg.types.CallbackQuery):
        await query.answer("All right, I learned a lot! Message me again sometime.")

    query_callbacks.inline_prompt(
        {
            "rate_not": CBCfg(rate_if_not, "◀"),
            "end_tour": CBCfg(end_tour, "■"),
            "rate_hot": CBCfg(rate_if_hot, "▶"),
        },
        reply_to=msg,
    )


@bot.on_message(filters=tg.filters.command(["walk"]))
async def start_tour(_: tg.Client, msg: tg.types.Message):
    if not user_preferences.prefs_for(msg.from_user).initialized():
        status = " ".join(
            """
            Looks like we don't have any data on your preferences! Send me a
            post or two, then start a tour with /walk 😊"
            """.split()
        )
        await msg.reply(status)
        return


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
