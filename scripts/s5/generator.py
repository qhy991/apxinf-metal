# Extracted verbatim from apxinf-metal scripts/bench_engines.sh (blob fd7de222).
# Two additions the harness supplies elsewhere in its own heredoc: the hashlib
# import, and CORPUS_SEED, which the caller sets before calling generate_prompt.
import hashlib

CORPUS_SEED = None
WORDS = """the of and to in a is that for it as was with be by on not he this are or from at which but have an
they you were her all she there would their we him been has when who will more no if out so said what up its
about into than them can only other new some could time these two may then do first any my now such like our
over man me even most made after also did many before must through back years where much your way well down
should because each just those people how too little state good very make world still own see men work long
get here between both life being under never day same another know while last might us great old year off
come since against go came right used take three states himself few house use during without again place
around however home small found thought went say part once general high upon school every don although got
united left number course war until always away something fact quite""".split()


def lcg(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF

    def step():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return state >> 33

    return step


def generate_prompt(case, chars_per_token=3.85):
    target_chars = max(24, int(round(case["target_tokens"] * chars_per_token)))
    case_seed = int(hashlib.sha256(f"{CORPUS_SEED}:{case['id']}".encode()).hexdigest()[:16], 16)
    step = lcg(case_seed)
    words = []
    length = 0
    while length < target_chars:
        word = WORDS[step() % len(WORDS)]
        words.append(word)
        length += len(word) + 1
    return " ".join(words)
