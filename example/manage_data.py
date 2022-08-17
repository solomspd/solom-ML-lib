from pathlib import Path
import random
import shutil
from tqdm import tqdm
import argparse as parse

num_test = 100

with Path.resolve(Path("../flower_photos")) as p:
	for j in p.iterdir():
		new_dir = p.parent / 'data' / j.name
		lst = sorted(j.iterdir())
		for k, f in tqdm(enumerate(lst[-100:]), leave=False):
			to = new_dir / "test" / f.name
			try:
				shutil.copy(f, to)
			except:
				to.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy(f, to)
		for k, f in tqdm(enumerate(lst[:-100]), leave=False):
			shutil.copy(f, new_dir / f.name)
