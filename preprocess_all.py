import os
import sys

import constants
from depparser import CharniakParser
from preprocessing import preprocess, StanfordCoreNLP

if __name__ == "__main__":
    constants.FLAG_ONTO = "wsj"
    constants.FLAG_DEPPARSER = "stdconv+charniak"

    if len(sys.argv) != 2:
        print("PYTHON2 preprocess_all.py INPUT_DIR")
        sys.exit(-1)
    INPUT_DIR = sys.argv[1]

    annotator = StanfordCoreNLP()
    annotator.setup()
    dparser = CharniakParser()

    for single_file in os.listdir(INPUT_DIR):
        if not single_file.endswith(".txt"): continue
        single_file = os.path.join(INPUT_DIR, single_file)
        instances = preprocess(single_file,
                               START_SNLP=True,
                               INPUT_AMR="sent",
                               PRP_FORMAT="plain",
                               proc1=annotator,
                               dparser=dparser)
