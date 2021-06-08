import datetime
import os
from urllib.request import urlopen

from tqdm import tqdm


def retrieve_archive(archive_type):
    with tqdm(unit_scale=True, unit_divisor=1024, unit="B") as progress:
        opath = f"./{archive_type}.csv.gz"
        if os.path.exists(opath):
            return
        date = (
            datetime.datetime.today() - datetime.datetime.timedelta(days=1)
        ).isoformat()
        archive = f"https://e621.net/db_export/{archive_type}-{date}"
        with urlopen(archive) as rsp:
            progress.total = int(rsp.headers["Content-Length"])
            progress.refresh()
            with open(opath, "wb+") as ostrm:
                while True:
                    chunk = rsp.read(4096)
                    if not chunk:
                        break
                    progress.update(len(chunk))
                    ostrm.write(chunk)


def main():
    retrieve_archive("posts")
    retrieve_archive("tags")


if __name__ == "__main__":
    main()
